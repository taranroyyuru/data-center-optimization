"""
One-time conversion script to convert KMZ/shapefile data to Parquet format for faster loading.
Run once: python convert_to_parquet.py

Geometry processing uses EPSG:5070 (US National Atlas Equal Area) for meter-based distances,
then converts back to EPSG:4326 for storage.
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

# Import utility (no Streamlit imports here)
from xlsx_utils import read_xlsx_sheet_no_openpyxl

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "datasets/cache"
CACHE_DIR.mkdir(exist_ok=True)

AQUEDUCT_CSV = ROOT / "datasets/Aqueduct40_waterrisk_download_Y2023M07D05/CVS/Aqueduct40_future_annual_y2023m07d05.csv"
AQUEDUCT_GDB = ROOT / "datasets/Aqueduct40_waterrisk_download_Y2023M07D05/GDB/Aq40_Y2023D07M05.gdb"
EGRID_XLSX = ROOT / "grid data/egrid2023_data_rev2 (2).xlsx"
EGRID_SUBREGION_KMZ = ROOT / "grid data/egrid2023_subregions.kmz"
URBAN_POLYGONS_SHP = ROOT / "urban population polygons/tl_2025_us_uac20.shp"

WATER_STRESS_CODE_FIELD = "bau50_ws_x_c"
WATER_STRESS_LABEL_FIELD = "bau50_ws_x_l"
US_MINX, US_MAXX, US_MINY, US_MAXY = -125.0, -66.5, 24.0, 49.8

# Projection for meter-based operations
METRIC_CRS = "EPSG:5070"  # US National Atlas Equal Area (meter-based)
GEO_CRS = "EPSG:4326"     # WGS84 (lon/lat)

def extract_subrgn_from_description(desc: str) -> str | None:
    import re
    match = re.search(r"Subregion\s*</td>\s*<td[^>]*>\s*([^<\s]+)", str(desc), flags=re.IGNORECASE)
    if not match:
        return None
    code = match.group(1).strip()
    if code in {"&lt;Null&gt;", "<Null>", "NULL", "null"}:
        return None
    return code

def convert_water_stress():
    """Convert water stress data to Parquet. Simplifies in projected CRS."""
    print("Converting water stress data...")
    
    water_attr = pd.read_csv(
        AQUEDUCT_CSV,
        usecols=["pfaf_id", WATER_STRESS_CODE_FIELD, WATER_STRESS_LABEL_FIELD],
    ).copy()
    water_attr["pfaf_id"] = water_attr["pfaf_id"].astype("int64")

    water_geom = gpd.read_file(AQUEDUCT_GDB, layer="future_annual")[["pfaf_id", "geometry"]].copy()
    water_geom["pfaf_id"] = water_geom["pfaf_id"].astype("int64")
    water_gdf = water_geom.merge(water_attr, on="pfaf_id", how="left")
    water_gdf = water_gdf.set_crs(GEO_CRS) if water_gdf.crs is None else water_gdf.to_crs(GEO_CRS)
    water_us = water_gdf.cx[US_MINX:US_MAXX, US_MINY:US_MAXY].copy()
    
    # Simplify in metric CRS (100m tolerance), then convert back to geographic
    water_us = water_us.to_crs(METRIC_CRS)
    water_us["geometry"] = water_us.geometry.simplify(100)  # 100 meters
    water_us = water_us.to_crs(GEO_CRS)
    
    water_us.to_parquet(CACHE_DIR / "water_stress.parquet")
    print(f"✓ Saved water stress to {CACHE_DIR / 'water_stress.parquet'}")


def convert_renewables():
    """Convert renewable energy regions to Parquet. Simplifies in projected CRS."""
    print("Converting renewable energy data...")
    
    sr = read_xlsx_sheet_no_openpyxl(EGRID_XLSX, "SRL23")[["SUBRGN", "SRNAME", "SRTRPR"]].copy()
    sr["SUBRGN"] = sr["SUBRGN"].astype(str).str.strip()
    sr["renewable_pct"] = pd.to_numeric(sr["SRTRPR"], errors="coerce") * 100.0

    subregion_gdf = gpd.read_file(EGRID_SUBREGION_KMZ)[["description", "geometry"]].copy()
    subregion_gdf = (
        subregion_gdf.set_crs(GEO_CRS) if subregion_gdf.crs is None else subregion_gdf.to_crs(GEO_CRS)
    )
    subregion_gdf["SUBRGN"] = subregion_gdf["description"].map(extract_subrgn_from_description)
    subregion_gdf = subregion_gdf.dropna(subset=["SUBRGN"])

    renewables_gdf = subregion_gdf.merge(sr[["SUBRGN", "SRNAME", "renewable_pct"]], on="SUBRGN", how="left")
    renewables_us = renewables_gdf.cx[US_MINX:US_MAXX, US_MINY:US_MAXY].copy()
    
    # Simplify in metric CRS (100m tolerance), then convert back to geographic
    renewables_us = renewables_us.to_crs(METRIC_CRS)
    renewables_us["geometry"] = renewables_us.geometry.simplify(100)  # 100 meters
    renewables_us = renewables_us.to_crs(GEO_CRS)
    
    renewables_us.to_parquet(CACHE_DIR / "renewables.parquet")
    print(f"✓ Saved renewables to {CACHE_DIR / 'renewables.parquet'}")


def convert_urban_zones():
    """Convert urban zones to Parquet (inverse). Buffers and simplifies in metric CRS."""
    print("Converting urban population polygons...")
    
    urban_gdf = gpd.read_file(URBAN_POLYGONS_SHP)[["geometry"]].copy()
    urban_gdf = urban_gdf.set_crs(GEO_CRS) if urban_gdf.crs is None else urban_gdf.to_crs(GEO_CRS)
    
    # Buffer urban zones by 1 mile (1609.34 m) in metric CRS
    urban_gdf = urban_gdf.to_crs(METRIC_CRS)
    urban_gdf["geometry"] = urban_gdf.geometry.buffer(1609.34)  # 1 mile in meters
    
    # Simplify in metric CRS (100m tolerance), then convert back to geographic
    urban_gdf["geometry"] = urban_gdf.geometry.simplify(100)  # 100 meters
    urban_gdf = urban_gdf.to_crs(GEO_CRS)
    
    urban_us = urban_gdf.cx[US_MINX:US_MAXX, US_MINY:US_MAXY].copy()
    
    # Create inverse: everything in the US EXCEPT buffered urban zones
    us_box = box(US_MINX, US_MINY, US_MAXX, US_MAXY)
    urban_union = urban_us.unary_union if len(urban_us) > 0 else None
    if urban_union is not None:
        urban_inverse_geom = us_box.difference(urban_union)
    else:
        urban_inverse_geom = us_box
    urban_non_zones_gdf = gpd.GeoDataFrame(geometry=[urban_inverse_geom], crs=GEO_CRS)
    
    urban_non_zones_gdf.to_parquet(CACHE_DIR / "urban_zones.parquet")
    print(f"✓ Saved urban zones (inverse) to {CACHE_DIR / 'urban_zones.parquet'}")

if __name__ == "__main__":
    print("Starting one-time Parquet conversion...\n")
    
    try:
        convert_water_stress()
        convert_renewables()
        convert_urban_zones()
        print("\n✓ All conversions complete! App will now load from Parquet files.")
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
