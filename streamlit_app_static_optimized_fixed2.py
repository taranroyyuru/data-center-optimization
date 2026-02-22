"""
Streamlit Data Center Siting Explorer – OPTIMIZED static PNG rendering.

Performance features:
- Rasterized masks (no per-cell intersection loops)
- Rasterized score grids (no spatial joins in render path)
- Precomputed all CRS conversions
- Direct PNG save (no temp files)
- Agg backend + non-interactive Matplotlib
- Disk-cached .npy arrays for masks and scores
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for speed
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import math
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D
from shapely.geometry import box

try:
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from xlsx_utils import read_xlsx_sheet_no_openpyxl

plt.ioff()  # Turn off interactive mode

# === SETUP ===
ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "datasets/cache"
PLOT_DIR = CACHE_DIR / "static_plots"
ARRAY_CACHE_DIR = CACHE_DIR / "arrays"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
ARRAY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

GEO_CRS = "EPSG:4326"
METRIC_CRS = "EPSG:5070"

AQUEDUCT_CSV = ROOT / "datasets/Aqueduct40_waterrisk_download_Y2023M07D05/CVS/Aqueduct40_future_annual_y2023m07d05.csv"
AQUEDUCT_GDB = ROOT / "datasets/Aqueduct40_waterrisk_download_Y2023M07D05/GDB/Aq40_Y2023D07M05.gdb"
PROJECTED_DC_DIR = ROOT / "datasets/im3_projected_data_centers"
EGRID_XLSX = ROOT / "grid data/egrid2023_data_rev2 (2).xlsx"
EGRID_SUBREGION_KMZ = ROOT / "grid data/egrid2023_subregions.kmz"
URBAN_POLYGONS_SHP = ROOT / "urban population polygons/tl_2025_us_uac20.shp"

WATER_STRESS_PARQUET = CACHE_DIR / "water_stress.parquet"
RENEWABLES_PARQUET = CACHE_DIR / "renewables.parquet"
URBAN_ZONES_PARQUET = CACHE_DIR / "urban_zones.parquet"

WATER_STRESS_CODE_FIELD = "bau50_ws_x_c"
WATER_STRESS_LABEL_FIELD = "bau50_ws_x_l"
US_MINX, US_MAXX, US_MINY, US_MAXY = -125.0, -66.5, 24.0, 49.8

# 5km grid in metric CRS bounds
GRID_MINX, GRID_MINY = 235129.0, 99999.0
GRID_MAXX, GRID_MAXY = 2963140.0, 3145314.0
CELL_SIZE = 5000

STRESS_LABELS = {
    -1: "Arid and low water use",
    0: "Low (<10%)",
    1: "Low-medium (10-20%)",
    2: "Medium-high (20-40%)",
    3: "High (40-80%)",
    4: "Extremely high (>80%)",
}
STRESS_COLORS = {
    -1: "#e8efe3", 0: "#b8e186", 1: "#7fbc41", 2: "#fdae61", 3: "#f46d43", 4: "#d73027",
}

RENEWABLE_LEVELS = [
    ("Very low (0-20%)", 0, 20),
    ("Low (20-35%)", 20, 35),
    ("Moderate (35-50%)", 35, 50),
    ("High (50-100%)", 50, 100.01),
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

WHITE_TO_GREEN = LinearSegmentedColormap.from_list("white_to_green", ["#ffffff", "#0b7d3e"])


# === RASTER GRID HELPERS ===
def get_grid_shape() -> tuple[int, int]:
    """Return (nrows, ncols) for the 5km grid in metric CRS (ceil to cover full extent)."""
    ncols = int(math.ceil((GRID_MAXX - GRID_MINX) / CELL_SIZE))
    nrows = int(math.ceil((GRID_MAXY - GRID_MINY) / CELL_SIZE))
    return nrows, ncols


def get_grid_bounds_aligned() -> tuple[float, float, float, float]:
    """Aligned metric bounds (minx, miny, maxx, maxy) that match the raster/grid shape."""
    nrows, ncols = get_grid_shape()
    maxx = GRID_MINX + ncols * CELL_SIZE
    maxy = GRID_MINY + nrows * CELL_SIZE
    return GRID_MINX, GRID_MINY, maxx, maxy


def get_grid_transform() -> Affine:
    """Return rasterio Affine transform for the aligned 5km grid in metric CRS."""
    minx, miny, maxx, maxy = get_grid_bounds_aligned()
    return Affine(CELL_SIZE, 0, minx, 0, -CELL_SIZE, maxy)

def rasterize_to_bool(geom, nrows: int, ncols: int, transform: Affine, fill: int = 0) -> np.ndarray:
    """Rasterize a geometry to boolean array (True where geom exists, False where not)."""
    if not HAS_RASTERIO:
        raise ImportError("rasterio required. Install: pip install rasterio")
    if geom is None or (hasattr(geom, "is_empty") and geom.is_empty):
        return np.zeros((nrows, ncols), dtype=bool)
    
    shapes = [(geom, 1)]
    raster = rasterize(shapes, out_shape=(nrows, ncols), transform=transform, fill=fill, dtype=np.uint8)
    return raster > 0


# === CORE HELPERS ===
def cached_png(path: Path, render_fn: Callable[[Path], None]) -> Path:
    """
    PNG cache pattern: If path exists, return it. Otherwise call render_fn(path) and return.
    render_fn receives the output path and must save figure to it.
    """
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    render_fn(path)
    return path


def extract_subrgn_from_description(desc: str) -> str | None:
    """Parse subregion code from KMZ description HTML."""
    match = re.search(r"Subregion\s*</td>\s*<td[^>]*>\s*([^<\s]+)", str(desc), flags=re.IGNORECASE)
    if not match:
        return None
    code = match.group(1).strip()
    return None if code in {"&lt;Null&gt;", "<Null>", "NULL", "null"} else code


def load_or_create_masks(us_boundary_metric, inverse_urban_metric) -> tuple[np.ndarray, np.ndarray]:
    """
    Load or create rasterized boolean masks for US boundary and allowed (non-urban) areas.
    Returns (mask_us, mask_allowed) as uint8 arrays.
    """
    mask_us_path = ARRAY_CACHE_DIR / "mask_us_5km.npy"
    mask_allowed_path = ARRAY_CACHE_DIR / "mask_allowed_5km.npy"
    
    nrows, ncols = get_grid_shape()
    transform = get_grid_transform()
    
    if mask_us_path.exists() and mask_allowed_path.exists():
        mask_us = np.load(mask_us_path)
        mask_allowed = np.load(mask_allowed_path)
        # If cached arrays were created with a different grid shape, regenerate them
        if mask_us.shape != (nrows, ncols) or mask_allowed.shape != (nrows, ncols):
            mask_us = rasterize_to_bool(us_boundary_metric, nrows, ncols, transform, fill=0).astype(np.uint8)
            mask_allowed = rasterize_to_bool(inverse_urban_metric, nrows, ncols, transform, fill=0).astype(np.uint8)
            np.save(mask_us_path, mask_us)
            np.save(mask_allowed_path, mask_allowed)

    else:
        mask_us = rasterize_to_bool(us_boundary_metric, nrows, ncols, transform, fill=0).astype(np.uint8)
        mask_allowed = rasterize_to_bool(inverse_urban_metric, nrows, ncols, transform, fill=0).astype(np.uint8)
        np.save(mask_us_path, mask_us)
        np.save(mask_allowed_path, mask_allowed)
    
    return mask_us, mask_allowed


def load_or_create_score_grids(water_metric: gpd.GeoDataFrame, renewables_metric: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Load or create score raster grids via rasterization+mapping (no spatial joins).
    Returns (water_score, renew_score) as float32 arrays in [0, 1] range.
    """
    water_score_path = ARRAY_CACHE_DIR / "water_score_5km.npy"
    renew_score_path = ARRAY_CACHE_DIR / "renew_score_5km.npy"
    
    nrows, ncols = get_grid_shape()
    transform = get_grid_transform()
    
    # Water stress: rasterize codes, map to scores
    if water_score_path.exists():
        water_score = np.load(water_score_path)
        if water_score.shape != (nrows, ncols):
            water_score_path.unlink(missing_ok=True)
            water_score = None

    else:
        # Build (geom, code) pairs
        shapes = [
            (row.geometry, int(row[WATER_STRESS_CODE_FIELD]) if pd.notna(row[WATER_STRESS_CODE_FIELD]) else -1)
            for _idx, row in water_metric.iterrows()
        ]
        code_grid = rasterize(shapes, out_shape=(nrows, ncols), transform=transform, fill=-1, dtype=np.float32)
        
        # Map codes to inverted scores (low stress = high score)
        stress_to_score = {-1: 0.0, 0: 1.0, 1: 0.75, 2: 0.5, 3: 0.25, 4: 0.0}
        water_score = np.zeros((nrows, ncols), dtype=np.float32)
        for code, score in stress_to_score.items():
            water_score[code_grid == code] = score
        np.save(water_score_path, water_score)
    
    # If water score cache was invalidated above, regenerate now
    if water_score is None:
        shapes = [
            (row.geometry, int(row[WATER_STRESS_CODE_FIELD]) if pd.notna(row[WATER_STRESS_CODE_FIELD]) else -1)
            for _idx, row in water_metric.iterrows()
        ]
        code_grid = rasterize(shapes, out_shape=(nrows, ncols), transform=transform, fill=-1, dtype=np.float32)
        stress_to_score = {-1: 0.0, 0: 1.0, 1: 0.75, 2: 0.5, 3: 0.25, 4: 0.0}
        water_score = np.zeros((nrows, ncols), dtype=np.float32)
        for code, score in stress_to_score.items():
            water_score[code_grid == code] = score
        np.save(water_score_path, water_score)

    # Renewables: rasterize percentages, map to score bins
    if renew_score_path.exists():
        renew_score = np.load(renew_score_path)
        if renew_score.shape != (nrows, ncols):
            renew_score_path.unlink(missing_ok=True)
            renew_score = None

    else:
        # Build (geom, pct) pairs
        shapes = [
            (row.geometry, float(row["renewable_pct"]) if pd.notna(row["renewable_pct"]) else 0.0)
            for _idx, row in renewables_metric.iterrows()
        ]
        pct_grid = rasterize(shapes, out_shape=(nrows, ncols), transform=transform, fill=0, dtype=np.float32)
        
        # Map percentages to score bins
        renew_score = np.zeros((nrows, ncols), dtype=np.float32)
        renew_score[(pct_grid >= 0) & (pct_grid < 20)] = 0.25
        renew_score[(pct_grid >= 20) & (pct_grid < 35)] = 0.5
        renew_score[(pct_grid >= 35) & (pct_grid < 50)] = 0.75
        renew_score[(pct_grid >= 50)] = 1.0
        np.save(renew_score_path, renew_score)
    
    if renew_score is None:
        shapes = [
            (row.geometry, float(row["renewable_pct"]) if pd.notna(row["renewable_pct"]) else 0.0)
            for _idx, row in renewables_metric.iterrows()
        ]
        pct_grid = rasterize(shapes, out_shape=(nrows, ncols), transform=transform, fill=0, dtype=np.float32)
        renew_score = np.zeros((nrows, ncols), dtype=np.float32)
        renew_score[(pct_grid >= 0) & (pct_grid < 20)] = 0.25
        renew_score[(pct_grid >= 20) & (pct_grid < 35)] = 0.5
        renew_score[(pct_grid >= 35) & (pct_grid < 50)] = 0.75
        renew_score[(pct_grid >= 50)] = 1.0
        np.save(renew_score_path, renew_score)

    return water_score, renew_score


@st.cache_data
def create_grid_5km() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Create 5km grid aligned to the raster masks/scores. Returns (grid_metric, grid_geo)."""
    nrows, ncols = get_grid_shape()
    minx, miny, maxx, maxy = get_grid_bounds_aligned()
    cells = []
    # Build in deterministic row-major order to match raster .ravel()
    for row in range(nrows):
        y0 = maxy - (row + 1) * CELL_SIZE
        y1 = maxy - row * CELL_SIZE
        for col in range(ncols):
            x0 = minx + col * CELL_SIZE
            x1 = minx + (col + 1) * CELL_SIZE
            cells.append({"geometry": box(x0, y0, x1, y1)})
    grid_metric = gpd.GeoDataFrame(cells, crs=METRIC_CRS)
    grid_geo = grid_metric.to_crs(GEO_CRS)
    return grid_metric, grid_geo

@st.cache_resource(show_spinner=False)
def load_context() -> dict:
    """Load and precompute all data: geometries, CRS variants, masks, scores."""
    use_parquet = all([WATER_STRESS_PARQUET.exists(), RENEWABLES_PARQUET.exists(), URBAN_ZONES_PARQUET.exists()])
    
    if use_parquet:
        water_us = gpd.read_parquet(WATER_STRESS_PARQUET)
        renewables_us = gpd.read_parquet(RENEWABLES_PARQUET)
        urban_non_zones_gdf = gpd.read_parquet(URBAN_ZONES_PARQUET)
    else:
        for path in [AQUEDUCT_CSV, AQUEDUCT_GDB, PROJECTED_DC_DIR, EGRID_XLSX, EGRID_SUBREGION_KMZ, URBAN_POLYGONS_SHP]:
            if not path.exists():
                raise FileNotFoundError(f"Missing: {path}")
        
        water_attr = pd.read_csv(AQUEDUCT_CSV, usecols=["pfaf_id", WATER_STRESS_CODE_FIELD, WATER_STRESS_LABEL_FIELD])
        water_attr["pfaf_id"] = water_attr["pfaf_id"].astype("int64")
        
        water_geom = gpd.read_file(AQUEDUCT_GDB, layer="future_annual")[["pfaf_id", "geometry"]]
        water_geom["pfaf_id"] = water_geom["pfaf_id"].astype("int64")
        water_gdf = water_geom.merge(water_attr, on="pfaf_id", how="left")
        water_us = water_gdf.set_crs(GEO_CRS) if water_gdf.crs is None else water_gdf.to_crs(GEO_CRS)
        water_us = water_us.cx[US_MINX:US_MAXX, US_MINY:US_MAXY].copy()
        water_us = water_us.to_crs(METRIC_CRS)
        water_us["geometry"] = water_us.geometry.simplify(100)
        water_us = water_us.to_crs(GEO_CRS)
        
        sr = read_xlsx_sheet_no_openpyxl(EGRID_XLSX, "SRL23")[["SUBRGN", "SRNAME", "SRTRPR"]]
        sr["SUBRGN"] = sr["SUBRGN"].astype(str).str.strip()
        sr["renewable_pct"] = pd.to_numeric(sr["SRTRPR"], errors="coerce") * 100.0
        
        subregion_gdf = gpd.read_file(EGRID_SUBREGION_KMZ)[["description", "geometry"]]
        subregion_gdf = subregion_gdf.set_crs(GEO_CRS) if subregion_gdf.crs is None else subregion_gdf.to_crs(GEO_CRS)
        subregion_gdf["SUBRGN"] = subregion_gdf["description"].map(extract_subrgn_from_description)
        subregion_gdf = subregion_gdf.dropna(subset=["SUBRGN"])
        
        renewables_gdf = subregion_gdf.merge(sr[["SUBRGN", "SRNAME", "renewable_pct"]], on="SUBRGN", how="left")
        renewables_us = renewables_gdf.cx[US_MINX:US_MAXX, US_MINY:US_MAXY].copy()
        renewables_us = renewables_us.to_crs(METRIC_CRS)
        renewables_us["geometry"] = renewables_us.geometry.simplify(100)
        renewables_us = renewables_us.to_crs(GEO_CRS)
        
        urban_gdf = gpd.read_file(URBAN_POLYGONS_SHP)[["geometry"]]
        urban_gdf = urban_gdf.set_crs(GEO_CRS) if urban_gdf.crs is None else urban_gdf.to_crs(GEO_CRS)
        urban_gdf_metric = urban_gdf.to_crs(METRIC_CRS)
        urban_gdf_metric["geometry"] = urban_gdf_metric.geometry.buffer(1609.34).simplify(100)
        urban_gdf = urban_gdf_metric.to_crs(GEO_CRS)
        urban_us = urban_gdf.cx[US_MINX:US_MAXX, US_MINY:US_MAXY].copy()
        
        us_box = box(US_MINX, US_MINY, US_MAXX, US_MAXY)
        urban_union = urban_us.unary_union if len(urban_us) > 0 else None
        urban_inverse_geom = us_box.difference(urban_union) if urban_union is not None else us_box
        urban_non_zones_gdf = gpd.GeoDataFrame(geometry=[urban_inverse_geom], crs=GEO_CRS)
    
    # Load data centers (read each file ONCE per projection/gravity)
    dc_by_proj_grav = {}
    for proj_dir in sorted([d for d in PROJECTED_DC_DIR.iterdir() if d.is_dir()]):
        proj = proj_dir.name
        by_grav = {}
        for fp in sorted(proj_dir.glob(f"{proj}_*_market_gravity.geojson")):
            grav = int(fp.stem.split("_")[-3])
            gdf = gpd.read_file(fp)
            gdf = gdf.set_crs(GEO_CRS) if gdf.crs is None else gdf.to_crs(GEO_CRS)
            by_grav[grav] = gdf
        if by_grav:
            dc_by_proj_grav[proj] = by_grav
    
    # Precompute all CRS variants to avoid re-conversions
    water_metric = water_us.to_crs(METRIC_CRS)
    renewables_metric = renewables_us.to_crs(METRIC_CRS)
    urban_metric = urban_non_zones_gdf.to_crs(METRIC_CRS)
    
    us_boundary_geo = renewables_us.dissolve().geometry.iloc[0]
    us_boundary_metric = renewables_metric.dissolve().geometry.iloc[0]
    
    inverse_urban_geo = urban_non_zones_gdf.geometry.iloc[0]
    inverse_urban_metric = urban_metric.geometry.iloc[0]
    
    # Precompute rasterized masks and score grids
    mask_us, mask_allowed = load_or_create_masks(us_boundary_metric, inverse_urban_metric)
    water_score, renew_score = load_or_create_score_grids(water_metric, renewables_metric)
    
    return {
        "water_us": water_us,
        "water_metric": water_metric,
        "renewables_us": renewables_us,
        "renewables_metric": renewables_metric,
        "urban_us": urban_non_zones_gdf,
        "urban_metric": urban_metric,
        "us_boundary_geo": us_boundary_geo,
        "us_boundary_metric": us_boundary_metric,
        "inverse_urban_geo": inverse_urban_geo,
        "inverse_urban_metric": inverse_urban_metric,
        "dc_by_projection_and_gravity": dc_by_proj_grav,
        "projection_values": sorted(dc_by_proj_grav.keys()),
        "gravity_values": sorted({g for by_g in dc_by_proj_grav.values() for g in by_g.keys()}),
        "mask_us": mask_us,
        "mask_allowed": mask_allowed,
        "water_score": water_score,
        "renew_score": renew_score,
    }


def render_background(
    background: str, projection: str, gravity: int, site_size: int, show_boundaries: bool, ctx: dict, out_path: Path
) -> None:
    """Render background to matplotlib figure and save directly to out_path."""
    dc_gdf = ctx["dc_by_projection_and_gravity"][projection][gravity].copy()
    dc_us = dc_gdf.cx[US_MINX:US_MAXX, US_MINY:US_MAXY].copy()
    sites = dc_us.copy()
    sites["geometry"] = sites.geometry.representative_point()
    
    fig, ax = plt.subplots(figsize=(14, 8.5), facecolor="white")
    ax.set_facecolor("white")
    
    if background == "Inverse Urban Zones":
        gpd.GeoSeries([ctx["us_boundary_geo"]], crs=GEO_CRS).plot(ax=ax, color="#0b7d3e", linewidth=0, alpha=1.0, zorder=1)
        if len(ctx["urban_us"]) > 0:
            ctx["urban_us"].plot(ax=ax, color="white", linewidth=0, alpha=1.0, zorder=2, rasterized=True)
            if show_boundaries:
                ctx["urban_us"].boundary.plot(ax=ax, color="#9ca3af", linewidth=0.25, alpha=0.7, zorder=3)
        gpd.GeoSeries([ctx["us_boundary_geo"]], crs=GEO_CRS).boundary.plot(ax=ax, color="#1f2937", linewidth=1.2, alpha=0.95, zorder=6)
    
    elif background == "Water stress":
        water_plot = ctx["water_us"].copy()
        water_plot["score"] = 1.0 - ((water_plot[WATER_STRESS_CODE_FIELD] + 1.0) / 5.0)
        water_plot.plot(ax=ax, column="score", cmap=WHITE_TO_GREEN, linewidth=0, alpha=1.0, zorder=2, rasterized=True, legend=False)
        if show_boundaries:
            water_plot.boundary.plot(ax=ax, color="#e5e7eb", linewidth=0.2, alpha=0.6, zorder=3)
        ctx["renewables_us"].dissolve().boundary.plot(ax=ax, color="#1f2937", linewidth=1.2, alpha=0.95, zorder=6)
        
        renew_handles = [
            Line2D([0], [0], marker="s", linestyle="None", markerfacecolor=STRESS_COLORS[c], markeredgecolor="none",
                   markersize=10, label=STRESS_LABELS[c])
            for c in [-1, 0, 1, 2, 3, 4]
        ]
        leg = ax.legend(handles=renew_handles, title="Water stress", loc="lower left", frameon=True, framealpha=0.94, fontsize=9)
        leg.set_zorder(1000)
    
    elif background == "Renewables %":
        renew_plot = ctx["renewables_us"].copy()
        renew_plot["score"] = (renew_plot["renewable_pct"] / 100.0).clip(0, 1)
        renew_plot.plot(ax=ax, column="score", cmap=WHITE_TO_GREEN, linewidth=0, alpha=1.0, zorder=2, rasterized=True, legend=False)
        if show_boundaries:
            renew_plot.boundary.plot(ax=ax, color="#e5e7eb", linewidth=0.25, alpha=0.7, zorder=3)
        ctx["renewables_us"].dissolve().boundary.plot(ax=ax, color="#1f2937", linewidth=1.25, alpha=0.95, zorder=6)
        
        renew_handles = [
            Line2D([0], [0], marker="s", linestyle="None", markerfacecolor=WHITE_TO_GREEN(i / 3.0),
                   markeredgecolor="none", markersize=10, label=label)
            for i, (label, _lo, _hi) in enumerate(RENEWABLE_LEVELS)
        ]
        leg = ax.legend(handles=renew_handles, title="Renewables %", loc="lower left", frameon=True, framealpha=0.94, fontsize=9)
        leg.set_zorder(1000)
    
    else:  # Hybrid: suitability grid
        _grid_metric, grid_geo = create_grid_5km()
        mask_us = ctx["mask_us"]
        mask_allowed = ctx["mask_allowed"]
        renew_score = ctx["renew_score"]
        water_score = ctx["water_score"]
        
        final_score = np.minimum(renew_score, water_score) * mask_us.astype(float) * mask_allowed.astype(float)
        score_flat = final_score.reshape(-1)
        valid_flat = score_flat > 0
        tiles = grid_geo.loc[valid_flat].copy()
        tiles["score"] = score_flat[valid_flat]
        tiles.plot(ax=ax, column="score", cmap=WHITE_TO_GREEN, linewidth=0, alpha=1.0, zorder=2, rasterized=True, legend=False)
        ctx["renewables_us"].dissolve().boundary.plot(ax=ax, color="#1f2937", linewidth=1.1, alpha=0.95, zorder=6)
    
    # Classify and plot sites
    sites["site_group"] = "Unknown"
    if background == "Inverse Urban Zones":
        sites["site_group"] = "Non-urban area"
    elif background in {"Water stress", "Hybrid"}:
        water_join = ctx["water_us"][[WATER_STRESS_CODE_FIELD, "geometry"]].dropna(subset=[WATER_STRESS_CODE_FIELD])
        sites = gpd.sjoin(sites, water_join, how="left", predicate="within")
        sites["site_group"] = sites[WATER_STRESS_CODE_FIELD].apply(
            lambda v: "Unknown" if pd.isna(v) else (
                "High/Extremely high stress" if int(v) >= 3 else ("Medium-high stress" if int(v) == 2 else "Low to low-medium stress")
            )
        )
    else:  # Renewables
        renew_join = ctx["renewables_us"][["renewable_pct", "geometry"]]
        sites = gpd.sjoin(sites, renew_join, how="left", predicate="within")
        sites["site_group"] = sites["renewable_pct"].apply(
            lambda v: "Unknown" if pd.isna(v) else (
                "High renewables (>=50%)" if v >= 50 else ("Medium renewables (30-50%)" if v >= 30 else "Low renewables (<30%)")
            )
        )
    
    for group_name in SITE_STYLES:
        group = sites[sites["site_group"] == group_name]
        if len(group) == 0:
            continue
        style = SITE_STYLES[group_name]
        group.plot(ax=ax, marker=style["marker"], color=style["color"], markersize=site_size,
                   edgecolor="white", linewidth=0.45, alpha=0.95, zorder=8)
    
    present_groups = [g for g in SITE_STYLES if g in set(sites["site_group"])]
    site_handles = [
        Line2D([0], [0], marker=SITE_STYLES[g]["marker"], linestyle="None",
               markerfacecolor=SITE_STYLES[g]["color"], markeredgecolor="white", markersize=9, label=g)
        for g in present_groups
    ]
    site_leg = ax.legend(handles=site_handles, title="Site classification", loc="upper left", frameon=True, framealpha=0.94, fontsize=9)
    site_leg.set_zorder(1000)
    
    ax.set_xlim(US_MINX, US_MAXX)
    ax.set_ylim(US_MINY, US_MAXY)
    ax.set_title(
        f"US {background} Background + Data Centers | {projection} | Gravity {gravity}",
        fontsize=13, pad=12,
    )
    ax.set_axis_off()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_heatmap(
    projection: str, gravity: int, site_size: int, selected_layers: list, ctx: dict, out_path: Path
) -> None:
    """Render heatmap with AND-logic. Saves directly to out_path. NO spatial joins."""
    dc_gdf = ctx["dc_by_projection_and_gravity"][projection][gravity].copy()
    dc_us = dc_gdf.cx[US_MINX:US_MAXX, US_MINY:US_MAXY].copy()
    
    _grid_metric, grid_geo = create_grid_5km()
    mask_us = ctx["mask_us"]
    mask_allowed = ctx["mask_allowed"]
    
    # Build final score from precomputed rasterized arrays (no sjoin!)
    final_score = np.ones(mask_us.size, dtype=float)
    if "Renewables" in selected_layers:
        final_score = np.minimum(final_score, ctx["renew_score"].flatten())
    if "Water Stress" in selected_layers:
        final_score = np.minimum(final_score, ctx["water_score"].flatten())
    
    final_score *= mask_us.flatten().astype(float) * mask_allowed.flatten().astype(float)
    final_score = final_score.reshape(mask_us.shape)
    
    score_flat = final_score.reshape(-1)
    valid_flat = score_flat > 0
    
    fig, ax = plt.subplots(figsize=(14, 8.5), facecolor="white")
    ax.set_facecolor("white")
    
    if valid_flat.sum() == 0:
        ax.text(0.5, 0.5, "No suitable cells", ha="center", va="center", transform=ax.transAxes, fontsize=14)
    else:
        tiles = grid_geo.loc[valid_flat].copy()
        tiles["score"] = score_flat[valid_flat]
        tiles.plot(ax=ax, column="score", cmap=WHITE_TO_GREEN, linewidth=0, alpha=1.0, zorder=2, rasterized=True, legend=False)
    
    ctx["renewables_us"].dissolve().boundary.plot(ax=ax, color="#1f2937", linewidth=1.1, alpha=0.95, zorder=6)
    
    if len(dc_us) > 0:
        pts = dc_us.copy()
        pts["geometry"] = pts.geometry.representative_point()
        pts.plot(ax=ax, marker="o", color="#007a6e", markersize=max(8, int(site_size * 0.6)),
                 edgecolor="white", linewidth=0.35, alpha=0.95, zorder=8)
    
    ax.set_xlim(US_MINX, US_MAXX)
    ax.set_ylim(US_MINY, US_MAXY)
    ax.set_title(
        f"Suitability Heatmap | {', '.join(selected_layers)} | {projection} | Gravity {gravity}",
        fontsize=13, pad=12,
    )
    ax.set_axis_off()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def get_or_render_background(
    background: str, projection: str, gravity: int, site_size: int, show_boundaries: bool, ctx: dict
) -> Path:
    """Get cached PNG or render it."""
    fname = f"bg__{background.lower().replace(' ', '_').replace('%', 'pct')}__{projection}__g{gravity}__s{site_size}__b{int(show_boundaries)}.png"
    path = PLOT_DIR / fname
    return cached_png(path, lambda p: render_background(background, projection, gravity, site_size, show_boundaries, ctx, p))


def get_or_render_heatmap(
    projection: str, gravity: int, site_size: int, selected_layers: list, ctx: dict
) -> Path:
    """Get cached PNG or render it."""
    layers_key = "-".join(sorted([s.lower().replace(" ", "_") for s in selected_layers]))
    fname = f"hm__{layers_key}__{projection}__g{gravity}__s{site_size}.png"
    path = PLOT_DIR / fname
    return cached_png(path, lambda p: render_heatmap(projection, gravity, site_size, selected_layers, ctx, p))


def pre_render_defaults(ctx: dict) -> None:
    """Pre-generate a small set of default images for snappy UI."""
    projs = ctx["projection_values"][:1] if len(ctx["projection_values"]) > 0 else []
    gravs = sorted(ctx["gravity_values"])[:2] if len(ctx["gravity_values"]) > 0 else []
    
    for proj in projs:
        for grav in gravs:
            for bg in ["Water stress", "Renewables %"]:
                _ = get_or_render_background(bg, proj, grav, 30, False, ctx)
            for layers in [["Water Stress", "Renewables"]]:
                _ = get_or_render_heatmap(proj, grav, 30, layers, ctx)


def main():
    st.set_page_config(page_title="Data Center Siting Explorer", layout="wide")
    st.title("Data Center Siting Explorer")
    st.caption("Static Matplotlib PNG views: water stress, renewables, urban zones, and suitability heatmaps.")
    
    try:
        ctx = load_context()
        st.session_state["_ctx_cache"] = ctx
        pre_render_defaults(ctx)
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        return
    
    with st.sidebar:
        st.header("Controls")
        view_mode = st.radio("View Mode", ["Single Background", "Multi-Layer Heatmap"], index=0)
        
        proj_default = "high_growth" if "high_growth" in ctx["projection_values"] else ctx["projection_values"][0]
        projection = st.selectbox("Projection", ctx["projection_values"], index=ctx["projection_values"].index(proj_default))
        grav_default = 25 if 25 in ctx["gravity_values"] else ctx["gravity_values"][0]
        gravity = st.select_slider("Market gravity", options=ctx["gravity_values"], value=grav_default)
        site_size = st.slider("Site size", min_value=12, max_value=80, value=30, step=2)
        
        if view_mode == "Single Background":
            background = st.radio("Background", ["Water stress", "Renewables %", "Inverse Urban Zones", "Hybrid"], index=0)
            show_boundaries = st.checkbox("Show boundaries", value=False)
        else:
            st.subheader("Suitability Heatmap")
            st.caption("AND-logic: green where both renewables high AND water stress low. Urban areas masked white.")
            selected_layers = st.multiselect(
                "Metric Layers",
                ["Water Stress", "Renewables"],
                default=["Water Stress", "Renewables"],
            )
    
    if view_mode == "Single Background":
        img_path = get_or_render_background(background, projection, gravity, site_size, show_boundaries, ctx)
        st.image(str(img_path), use_container_width=True)
    else:
        if not selected_layers:
            st.warning("Please select at least one metric layer.")
        else:
            img_path = get_or_render_heatmap(projection, gravity, site_size, selected_layers, ctx)
            st.image(str(img_path), use_container_width=True)


if __name__ == "__main__":
    main()