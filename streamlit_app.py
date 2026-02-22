from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent

AQUEDUCT_CSV = ROOT / "datasets/Aqueduct40_waterrisk_download_Y2023M07D05/CVS/Aqueduct40_future_annual_y2023m07d05.csv"
AQUEDUCT_GDB = ROOT / "datasets/Aqueduct40_waterrisk_download_Y2023M07D05/GDB/Aq40_Y2023D07M05.gdb"
PROJECTED_DC_DIR = ROOT / "datasets/im3_projected_data_centers"
EGRID_XLSX = ROOT / "grid data/egrid2023_data_rev2 (2).xlsx"
EGRID_SUBREGION_KMZ = ROOT / "grid data/egrid2023_subregions.kmz"

WATER_STRESS_CODE_FIELD = "bau50_ws_x_c"
WATER_STRESS_LABEL_FIELD = "bau50_ws_x_l"

US_MINX, US_MAXX, US_MINY, US_MAXY = -125.0, -66.5, 24.0, 49.8

STRESS_ORDER = [-1, 0, 1, 2, 3, 4]
STRESS_LABELS = {
    -1: "Arid and low water use",
    0: "Low (<10%)",
    1: "Low-medium (10-20%)",
    2: "Medium-high (20-40%)",
    3: "High (40-80%)",
    4: "Extremely high (>80%)",
}
STRESS_COLORS = {
    -1: "#e8efe3",
    0: "#b8e186",
    1: "#7fbc41",
    2: "#fdae61",
    3: "#f46d43",
    4: "#d73027",
}

RENEWABLE_LEVELS = [
    ("Very low (0-20%)", 0, 20, "#f7fcf5"),
    ("Low (20-35%)", 20, 35, "#c7e9c0"),
    ("Moderate (35-50%)", 35, 50, "#74c476"),
    ("High (50-100%)", 50, 100.01, "#238b45"),
]
RENEWABLE_DOT_LEVELS = [
    ("Very low (0-20%)", 0, 20, "......"),
    ("Low (20-35%)", 20, 35, "...."),
    ("Moderate (35-50%)", 35, 50, ".."),
    ("High (50-100%)", 50, 100.01, "."),
]

SITE_STYLES = {
    "Low to low-medium stress": {"marker": "o", "color": "#0f766e"},
    "Medium-high stress": {"marker": "^", "color": "#b45309"},
    "High/Extremely high stress": {"marker": "X", "color": "#b91c1c"},
    "Low renewables (<30%)": {"marker": "X", "color": "#b91c1c"},
    "Medium renewables (30-50%)": {"marker": "^", "color": "#b45309"},
    "High renewables (>=50%)": {"marker": "o", "color": "#0f766e"},
    "Unknown": {"marker": "s", "color": "#6b7280"},
}


def _col_to_idx(col_ref: str) -> int:
    n = 0
    for ch in col_ref:
        if ch.isalpha():
            n = n * 26 + (ord(ch.upper()) - 64)
    return n - 1


def read_xlsx_sheet_no_openpyxl(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    ns = {
        "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }

    with zipfile.ZipFile(xlsx_path) as zf:
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall("{http://schemas.openxmlformats.org/package/2006/relationships}Relationship")
        }

        sheet_rid = None
        for sh in wb.findall("m:sheets/m:sheet", ns):
            if sh.attrib["name"] == sheet_name:
                sheet_rid = sh.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
                break
        if sheet_rid is None:
            raise KeyError(f"Sheet {sheet_name!r} not found in {xlsx_path.name}")

        target = rel_map[sheet_rid]
        if not target.startswith("xl/"):
            target = f"xl/{target}"

        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            sroot = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in sroot.findall("m:si", ns):
                shared.append("".join((t.text or "") for t in si.findall(".//m:t", ns)))

        sroot = ET.fromstring(zf.read(target))
        rows = []
        for row in sroot.findall(".//m:sheetData/m:row", ns):
            rec = {}
            for c in row.findall("m:c", ns):
                ref = c.attrib.get("r", "A1")
                idx = _col_to_idx("".join(ch for ch in ref if ch.isalpha()))
                v = c.find("m:v", ns)
                if v is None or v.text is None:
                    rec[idx] = ""
                    continue
                val = v.text
                if c.attrib.get("t") == "s":
                    val = shared[int(val)]
                rec[idx] = val
            rows.append(rec)

    width = max(max(row.keys(), default=0) for row in rows) + 1
    matrix = [[row.get(i, "") for i in range(width)] for row in rows]
    return pd.DataFrame(matrix[2:], columns=matrix[1])


def extract_subrgn_from_description(desc: str) -> str | None:
    match = re.search(r"Subregion\s*</td>\s*<td[^>]*>\s*([^<\s]+)", str(desc), flags=re.IGNORECASE)
    if not match:
        return None
    code = match.group(1).strip()
    if code in {"&lt;Null&gt;", "<Null>", "NULL", "null"}:
        return None
    return code


@st.cache_resource(show_spinner=False)
def load_context() -> dict:
    for path in [AQUEDUCT_CSV, AQUEDUCT_GDB, PROJECTED_DC_DIR, EGRID_XLSX, EGRID_SUBREGION_KMZ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")

    water_attr = pd.read_csv(
        AQUEDUCT_CSV,
        usecols=["pfaf_id", WATER_STRESS_CODE_FIELD, WATER_STRESS_LABEL_FIELD],
    ).copy()
    water_attr["pfaf_id"] = water_attr["pfaf_id"].astype("int64")

    water_geom = gpd.read_file(AQUEDUCT_GDB, layer="future_annual")[["pfaf_id", "geometry"]].copy()
    water_geom["pfaf_id"] = water_geom["pfaf_id"].astype("int64")
    water_gdf = water_geom.merge(water_attr, on="pfaf_id", how="left")
    water_gdf = water_gdf.set_crs("EPSG:4326") if water_gdf.crs is None else water_gdf.to_crs("EPSG:4326")
    water_us = water_gdf.cx[US_MINX:US_MAXX, US_MINY:US_MAXY].copy()
    us_outline = water_us.dissolve().boundary

    projection_dirs = sorted([d for d in PROJECTED_DC_DIR.iterdir() if d.is_dir()])
    dc_by_projection_and_gravity = {}
    for proj_dir in projection_dirs:
        projection = proj_dir.name
        geojson_files = sorted(proj_dir.glob(f"{projection}_*_market_gravity.geojson"))
        by_gravity = {}
        for fp in geojson_files:
            gravity = int(fp.stem.split("_")[-3])
            gdf = gpd.read_file(fp)
            gdf = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf.to_crs("EPSG:4326")
            by_gravity[gravity] = gdf
        if by_gravity:
            dc_by_projection_and_gravity[projection] = by_gravity

    sr = read_xlsx_sheet_no_openpyxl(EGRID_XLSX, "SRL23")[["SUBRGN", "SRNAME", "SRTRPR"]].copy()
    sr["SUBRGN"] = sr["SUBRGN"].astype(str).str.strip()
    sr["renewable_pct"] = pd.to_numeric(sr["SRTRPR"], errors="coerce") * 100.0

    subregion_gdf = gpd.read_file(EGRID_SUBREGION_KMZ)[["description", "geometry"]].copy()
    subregion_gdf = (
        subregion_gdf.set_crs("EPSG:4326") if subregion_gdf.crs is None else subregion_gdf.to_crs("EPSG:4326")
    )
    subregion_gdf["SUBRGN"] = subregion_gdf["description"].map(extract_subrgn_from_description)
    subregion_gdf = subregion_gdf.dropna(subset=["SUBRGN"])

    renewables_gdf = subregion_gdf.merge(sr[["SUBRGN", "SRNAME", "renewable_pct"]], on="SUBRGN", how="left")
    renewables_us = renewables_gdf.cx[US_MINX:US_MAXX, US_MINY:US_MAXY].copy()
    us_outline_renewables = renewables_us.dissolve().boundary

    return {
        "water_us": water_us,
        "us_outline": us_outline,
        "renewables_us": renewables_us,
        "us_outline_renewables": us_outline_renewables,
        "dc_by_projection_and_gravity": dc_by_projection_and_gravity,
        "projection_values": sorted(dc_by_projection_and_gravity.keys()),
        "gravity_values": sorted({g for by_g in dc_by_projection_and_gravity.values() for g in by_g.keys()}),
    }


def classify_sites_background(background: str, sites: gpd.GeoDataFrame, ctx: dict) -> tuple[gpd.GeoDataFrame, list, str]:
    if background in {"Water stress", "Hybrid"}:
        water_us = ctx["water_us"]
        water_join = water_us[[WATER_STRESS_CODE_FIELD, WATER_STRESS_LABEL_FIELD, "geometry"]].dropna(
            subset=[WATER_STRESS_CODE_FIELD]
        )
        sites = gpd.sjoin(sites, water_join, how="left", predicate="within")
        sites = sites.rename(columns={WATER_STRESS_CODE_FIELD: "site_metric"})

        def site_group(v):
            if pd.isna(v):
                return "Unknown"
            v = int(v)
            if v >= 3:
                return "High/Extremely high stress"
            if v == 2:
                return "Medium-high stress"
            return "Low to low-medium stress"

        sites["site_group"] = sites["site_metric"].apply(site_group)
        bg_handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="None",
                markerfacecolor=STRESS_COLORS[c],
                markeredgecolor="none",
                markersize=10,
                label=STRESS_LABELS[c],
            )
            for c in STRESS_ORDER
        ]
        return sites, bg_handles, "Water stress levels"

    renewables_us = ctx["renewables_us"]
    sites = gpd.sjoin(sites, renewables_us[["renewable_pct", "geometry"]], how="left", predicate="within")

    def site_group(v):
        if pd.isna(v):
            return "Unknown"
        if v >= 50:
            return "High renewables (>=50%)"
        if v >= 30:
            return "Medium renewables (30-50%)"
        return "Low renewables (<30%)"

    sites["site_group"] = sites["renewable_pct"].apply(site_group)
    bg_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=10,
            label=label,
        )
        for label, _lo, _hi, color in RENEWABLE_LEVELS
    ]
    return sites, bg_handles, "Renewables levels"


def make_figure(background: str, projection: str, gravity: int, site_size: int, show_boundaries: bool, ctx: dict):
    dc_gdf = ctx["dc_by_projection_and_gravity"][projection][gravity].copy()
    dc_us = dc_gdf.cx[US_MINX:US_MAXX, US_MINY:US_MAXY].copy()
    sites = dc_us.copy()
    sites["geometry"] = sites.geometry.representative_point()

    fig, ax = plt.subplots(figsize=(14, 8.5), facecolor="#f7f6f2")

    if background in {"Water stress", "Hybrid"}:
        ax.set_facecolor("#eaf2f7")
        water_us = ctx["water_us"]
        for code in STRESS_ORDER:
            subset = water_us[water_us[WATER_STRESS_CODE_FIELD] == code]
            if len(subset) > 0:
                subset.plot(ax=ax, color=STRESS_COLORS[code], linewidth=0, alpha=0.85)
        if show_boundaries:
            water_us.boundary.plot(ax=ax, color="#ffffff", linewidth=0.15, alpha=0.35)
        ctx["us_outline"].plot(ax=ax, color="#1f2937", linewidth=1.2, alpha=0.95, zorder=6)

        if background == "Hybrid":
            renewables_us = ctx["renewables_us"]
            renewables_plot = renewables_us.copy()
            renewables_plot["renew_level"] = pd.cut(
                renewables_plot["renewable_pct"],
                bins=[0, 20, 35, 50, 100.01],
                labels=[lvl[0] for lvl in RENEWABLE_DOT_LEVELS],
                include_lowest=True,
                right=False,
            )
            for label, _lo, _hi, hatch in RENEWABLE_DOT_LEVELS:
                subset = renewables_plot[renewables_plot["renew_level"] == label]
                if len(subset) > 0:
                    subset.plot(
                        ax=ax,
                        facecolor="none",
                        edgecolor="#7a7a7a",
                        linewidth=0.25,
                        alpha=0.9,
                        hatch=hatch,
                        zorder=7,
                    )
    else:
        ax.set_facecolor("#ecf7f2")
        renewables_us = ctx["renewables_us"]
        renewables_plot = renewables_us.copy()
        renewables_plot["renew_level"] = pd.cut(
            renewables_plot["renewable_pct"],
            bins=[0, 20, 35, 50, 100.01],
            labels=[lvl[0] for lvl in RENEWABLE_LEVELS],
            include_lowest=True,
            right=False,
        )
        for label, _lo, _hi, color in RENEWABLE_LEVELS:
            subset = renewables_plot[renewables_plot["renew_level"] == label]
            if len(subset) > 0:
                subset.plot(ax=ax, color=color, linewidth=0, alpha=0.9)
        if show_boundaries:
            renewables_us.boundary.plot(ax=ax, color="white", linewidth=0.45, alpha=0.9, zorder=5)
        ctx["us_outline_renewables"].plot(ax=ax, color="#1f2937", linewidth=1.25, alpha=0.95, zorder=6)

    sites, bg_handles, bg_title = classify_sites_background(background, sites, ctx)
    present_groups = [g for g in SITE_STYLES if g in set(sites["site_group"])]
    for group_name in present_groups:
        style = SITE_STYLES[group_name]
        group = sites[sites["site_group"] == group_name]
        if len(group) == 0:
            continue
        group.plot(
            ax=ax,
            marker=style["marker"],
            color=style["color"],
            markersize=site_size,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.95,
            zorder=8,
        )

    site_handles = [
        Line2D(
            [0],
            [0],
            marker=SITE_STYLES[k]["marker"],
            linestyle="None",
            markerfacecolor=SITE_STYLES[k]["color"],
            markeredgecolor="white",
            markersize=9,
            label=k,
        )
        for k in present_groups
    ]
    site_legend = ax.legend(
        handles=site_handles,
        title="Site classification",
        loc="upper left",
        frameon=True,
        framealpha=0.94,
        fontsize=9,
        title_fontsize=10,
    )
    site_legend.set_zorder(1000)
    ax.add_artist(site_legend)
    if bg_handles:
        leg_bg = ax.legend(
            handles=bg_handles,
            title=bg_title,
            loc="lower left",
            frameon=True,
            framealpha=0.94,
            fontsize=9,
            title_fontsize=10,
        )
        leg_bg.set_zorder(1000)
        ax.add_artist(leg_bg)

    if background == "Hybrid":
        hatch_handles = [
            Patch(facecolor="white", edgecolor="#7a7a7a", hatch=hatch, label=label)
            for label, _lo, _hi, hatch in RENEWABLE_DOT_LEVELS
        ]
        leg_hatch = ax.legend(
            handles=hatch_handles,
            title="Renewables dot density",
            loc="lower right",
            frameon=True,
            framealpha=0.94,
            fontsize=9,
            title_fontsize=10,
        )
        leg_hatch.set_zorder(1000)
        ax.add_artist(leg_hatch)

    ax.set_xlim(US_MINX, US_MAXX)
    ax.set_ylim(US_MINY, US_MAXY)
    ax.set_title(
        f"US {background} Background + Proposed Data Centers | Projection: {projection} | Market Gravity {gravity}",
        fontsize=13,
        pad=12,
    )
    ax.set_axis_off()
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="Data Center Siting Explorer", layout="wide")
    st.title("Data Center Siting Explorer")
    st.caption("Toggle water stress vs renewable subregion background and compare projected siting outputs.")

    try:
        ctx = load_context()
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        return

    with st.sidebar:
        st.header("Controls")
        background = st.radio("Background", ["Water stress", "Renewables %", "Hybrid"], index=0)
        projection_default = "high_growth" if "high_growth" in ctx["projection_values"] else ctx["projection_values"][0]
        projection = st.selectbox("Projection", ctx["projection_values"], index=ctx["projection_values"].index(projection_default))
        gravity_default = 25 if 25 in ctx["gravity_values"] else ctx["gravity_values"][0]
        gravity = st.select_slider("Market gravity", options=ctx["gravity_values"], value=gravity_default)
        site_size = st.slider("Site size", min_value=12, max_value=80, value=30, step=2)
        show_boundaries = st.checkbox("Show boundaries", value=True)

    fig = make_figure(background, projection, gravity, site_size, show_boundaries, ctx)
    st.pyplot(fig, clear_figure=True)


if __name__ == "__main__":
    main()
