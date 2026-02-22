from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import box

# Import utility for XLSX reading (no Streamlit imports there)
from xlsx_utils import read_xlsx_sheet_no_openpyxl


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "datasets/cache"

# CRS settings
GEO_CRS = "EPSG:4326"     # WGS84 (lon/lat, for storage and display)
METRIC_CRS = "EPSG:5070"  # US National Atlas Equal Area (for distances and operations)

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

URBAN_ZONES_COLOR = "#a78bfa"
URBAN_ZONES_ALPHA = 0.5

SITE_STYLES = {
    "Low to low-medium stress": {"marker": "o", "color": "#0f766e"},
    "Medium-high stress": {"marker": "^", "color": "#b45309"},
    "High/Extremely high stress": {"marker": "X", "color": "#b91c1c"},
    "Low renewables (<30%)": {"marker": "X", "color": "#b91c1c"},
    "Medium renewables (30-50%)": {"marker": "^", "color": "#b45309"},
    "High renewables (>=50%)": {"marker": "o", "color": "#0f766e"},
    "Unknown": {"marker": "s", "color": "#6b7280"},
}


def extract_subrgn_from_description(desc: str) -> str | None:
    """Extract subregion code from KMZ description HTML."""
    match = re.search(r"Subregion\s*</td>\s*<td[^>]*>\s*([^<\s]+)", str(desc), flags=re.IGNORECASE)
    if not match:
        return None
    code = match.group(1).strip()
    if code in {"&lt;Null&gt;", "<Null>", "NULL", "null"}:
        return None
    return code


@st.cache_data
def create_cached_grid(cell_size_degrees: float = 0.15) -> tuple[gpd.GeoDataFrame, np.ndarray, np.ndarray]:
    """
    Create grid and precompute centroids in metric CRS.
    
    Grid cells are defined in EPSG:4326, but centroids are computed in EPSG:5070
    (meter-based), then converted back to lon/lat for pydeck compatibility.
    
    Returns:
        (grid GeoDataFrame in EPSG:4326, longitudes array, latitudes array)
    """
    cells = []
    for x in np.arange(US_MINX, US_MAXX, cell_size_degrees):
        for y in np.arange(US_MINY, US_MAXY, cell_size_degrees):
            cell = box(x, y, x + cell_size_degrees, y + cell_size_degrees)
            cells.append({"geometry": cell})
    grid = gpd.GeoDataFrame(cells, crs=GEO_CRS)
    
    # Compute centroids in metric CRS for proper geospatial calculations
    grid_metric = grid.to_crs(METRIC_CRS)
    centroids_metric = grid_metric.geometry.centroid
    centroids_geo = gpd.GeoSeries(centroids_metric, crs=METRIC_CRS).to_crs(GEO_CRS)
    
    lons = centroids_geo.x.values
    lats = centroids_geo.y.values
    
    return grid, lons, lats


@st.cache_data
def create_grid_5km() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Create a 5-kilometer grid (5000m x 5000m cells) for the heatmap.
    
    Grid is built in EPSG:5070 (projected, meter-based) for accuracy,
    then converted to EPSG:4326 for pydeck rendering.
    
    Returns:
        (grid_metric: GeoDataFrame in EPSG:5070,
         grid_geo: GeoDataFrame in EPSG:4326,
         centroids_metric: GeoDataFrame in EPSG:5070,
         centroids_geo: GeoDataFrame in EPSG:4326)
    """
    from shapely.geometry import Point
    
    # Use US boundary (EPSG:5070) to build grid
    us_minx, us_miny = 235129.0, 99999.0    # Approximate bounds in EPSG:5070
    us_maxx, us_maxy = 2963140.0, 3145314.0  # (tighter than bbox from degree bounds)
    
    cells_metric = []
    
    # Iterate in 5000m steps (EPSG:5070 is in meters)
    x = us_minx
    while x < us_maxx:
        y = us_miny
        while y < us_maxy:
            cell = box(x, y, x + 5000, y + 5000)
            cells_metric.append({"geometry": cell})
            y += 5000
        x += 5000
    
    # Create grid in metric CRS
    grid_metric = gpd.GeoDataFrame(cells_metric, crs=METRIC_CRS)
    
    # Convert to geographic CRS for pydeck
    grid_geo = grid_metric.to_crs(GEO_CRS)
    
    # Compute centroids in both CRS
    centroids_metric = gpd.GeoDataFrame(
        geometry=grid_metric.geometry.centroid,
        crs=METRIC_CRS,
        index=grid_metric.index
    )
    centroids_geo = centroids_metric.to_crs(GEO_CRS)
    
    return grid_metric, grid_geo, centroids_metric, centroids_geo


def compute_us_mask_area_based(
    grid_metric: gpd.GeoDataFrame,
    renewables_us: gpd.GeoDataFrame,
    overlap_threshold: float = 0.95
) -> np.ndarray:
    """
    Compute area-based US boundary mask using overlap fraction.
    
    A grid cell is considered "in US" if at least overlap_threshold (95%)
    of the cell area overlaps with the dissolved renewables boundary.
    This prevents edge slivers from leaking outside the US outline.
    
    Args:
        grid_metric: Grid GeoDataFrame in EPSG:5070
        renewables_us: Renewables GeoDataFrame in EPSG:4326
        overlap_threshold: Minimum overlap fraction (0..1) to include cell (default 0.95)
    
    Returns:
        Boolean array of len(grid), True if cell sufficiently overlaps US boundary
    """
    if len(renewables_us) == 0:
        return np.ones(len(grid_metric), dtype=bool)
    
    try:
        # Convert renewables to metric CRS for area calculations
        renewables_metric = renewables_us.to_crs(METRIC_CRS)
        us_boundary = renewables_metric.dissolve().geometry.iloc[0]
        
        # Repair any invalid geometries
        grid_metric_valid = grid_metric.copy()
        grid_metric_valid["geometry"] = grid_metric_valid.geometry.buffer(0)
        if not us_boundary.is_valid:
            from shapely.ops import unary_union
            us_boundary = unary_union(renewables_metric.geometry)
        
        # Compute overlap fraction for each cell
        mask = np.zeros(len(grid_metric), dtype=bool)
        cell_areas = grid_metric_valid.geometry.area.values
        
        for idx, cell in enumerate(grid_metric_valid.geometry):
            try:
                # Intersection area
                overlap_geom = cell.intersection(us_boundary)
                overlap_area = overlap_geom.area
                overlap_fraction = overlap_area / cell_areas[idx] if cell_areas[idx] > 0 else 0
                mask[idx] = overlap_fraction >= overlap_threshold
            except Exception:
                mask[idx] = False
        
        return mask
    except Exception as e:
        st.warning(f"Error computing US area-based mask: {e}")
        return np.ones(len(grid_metric), dtype=bool)


def compute_allowed_mask_area_based(
    grid_metric: gpd.GeoDataFrame,
    inverse_urban_geom: object,
    overlap_threshold: float = 0.95
) -> np.ndarray:
    """
    Compute area-based allowed (non-urban) mask using overlap fraction.
    
    A grid cell is considered "allowed" (buildable) if at least overlap_threshold (95%)
    of the cell area is within the inverse-urban polygon (i.e., not in urban zone).
    This prevents urban areas from leaking color into the heatmap.
    
    Args:
        grid_metric: Grid GeoDataFrame in EPSG:5070
        inverse_urban_geom: Geometric object (polygon) representing allowed areas, in EPSG:5070
        overlap_threshold: Minimum in-zone fraction (0..1) to allow cell (default 0.95)
    
    Returns:
        Boolean array of len(grid), True if cell is sufficiently non-urban
    """
    if inverse_urban_geom is None or inverse_urban_geom.is_empty:
        return np.ones(len(grid_metric), dtype=bool)
    
    try:
        # Repair geometries
        grid_metric_valid = grid_metric.copy()
        grid_metric_valid["geometry"] = grid_metric_valid.geometry.buffer(0)
        if not inverse_urban_geom.is_valid:
            inverse_urban_geom = inverse_urban_geom.buffer(0)
        
        # Compute in-zone fraction for each cell
        mask = np.zeros(len(grid_metric), dtype=bool)
        cell_areas = grid_metric_valid.geometry.area.values
        
        for idx, cell in enumerate(grid_metric_valid.geometry):
            try:
                # Intersection with allowed area
                allowed_geom = cell.intersection(inverse_urban_geom)
                allowed_area = allowed_geom.area
                allowed_fraction = allowed_area / cell_areas[idx] if cell_areas[idx] > 0 else 0
                mask[idx] = allowed_fraction >= overlap_threshold
            except Exception:
                mask[idx] = False
        
        return mask
    except Exception as e:
        st.warning(f"Error computing allowed area mask: {e}")
        return np.ones(len(grid_metric), dtype=bool)





def compute_renewables_score(
    centroids_metric: gpd.GeoDataFrame,
    renewables_us: gpd.GeoDataFrame,
    grid_metric: gpd.GeoDataFrame
) -> np.ndarray:
    """
    Compute renewable energy contribution scores for grid cells based on renewable_pct binning.
    
    Scoring bins (same as RENEWABLE_LEVELS):
    - Very low (0-20%): score 0.25
    - Low (20-35%): score 0.5
    - Moderate (35-50%): score 0.75
    - High (50-100%): score 1.0
    
    Cells outside any renewable region get score 0.0.
    
    Args:
        centroids_metric: Grid centroids in EPSG:5070
        renewables_us: Renewables GeoDataFrame with renewable_pct column (in EPSG:4326)
        grid_metric: Grid GeoDataFrame in EPSG:5070 (for index alignment)
    
    Returns:
        Float array [0, 1] of len(grid)
    """
    score = np.zeros(len(grid_metric), dtype=float)
    
    if len(renewables_us) == 0:
        return score
    
    try:
        # Convert renewables to metric CRS for spatial join
        renewables_metric = renewables_us.to_crs(METRIC_CRS)
        
        # Spatial join: grid centroids to renewables polygons to get renewable_pct
        joined = gpd.sjoin(
            centroids_metric,
            renewables_metric[["renewable_pct", "geometry"]],
            how="left",
            predicate="within"
        )
        
        # Get renewable_pct; handle multiple matches by taking first
        grouped = joined.groupby(joined.index)["renewable_pct"].first()
        pct_values = grouped.values
        
        # Assign scores based on bins
        score_array = np.zeros(len(pct_values), dtype=float)
        pct = pct_values
        
        score_array[(pct >= 0) & (pct < 20)] = 0.25
        score_array[(pct >= 20) & (pct < 35)] = 0.5
        score_array[(pct >= 35) & (pct < 50)] = 0.75
        score_array[(pct >= 50) & (pct <= 100.01)] = 1.0
        score_array[pd.isna(pct)] = 0.0
        
        # Map back to original grid order
        score_series = pd.Series(score_array, index=grouped.index)
        score = score_series.reindex(grid_metric.index, fill_value=0.0).values
        
        return score
    except Exception as e:
        st.warning(f"Error computing renewables score: {e}")
        return score


def compute_water_stress_score(
    centroids_metric: gpd.GeoDataFrame,
    water_us: gpd.GeoDataFrame,
    grid_metric: gpd.GeoDataFrame
) -> np.ndarray:
    """
    Compute water stress contribution scores for grid cells.
    
    Scoring (low stress = high score, high stress = low score):
    - Arid/low water use (-1): 0.0 (white, not buildable)
    - Low (<10%) (0): 1.0 (full green, optimal)
    - Low-medium (10-20%) (1): 0.75 (good)
    - Medium-high (20-40%) (2): 0.5 (neutral)
    - High (40-80%) (3): 0.25 (poor)
    - Extremely high (>80%) (4): 0.0 (white, not buildable)
    
    Cells outside any water stress region get score 0.0 (conservative default).
    
    Args:
        centroids_metric: Grid centroids in EPSG:5070
        water_us: Water stress GeoDataFrame with bau50_ws_x_c (stress code) column (in EPSG:4326)
        grid_metric: Grid GeoDataFrame in EPSG:5070 (for index alignment)
    
    Returns:
        Float array [0, 1] of len(grid)
    """
    score = np.zeros(len(grid_metric), dtype=float)
    
    if len(water_us) == 0:
        return score
    
    try:
        # Convert water to metric CRS for spatial join
        water_metric = water_us.to_crs(METRIC_CRS)
        
        # Spatial join: grid centroids to water stress polygons to get stress code
        joined = gpd.sjoin(
            centroids_metric,
            water_metric[[WATER_STRESS_CODE_FIELD, "geometry"]],
            how="left",
            predicate="within"
        )
        
        # Get stress codes; handle multiple matches by taking first
        stress_codes = joined.groupby(joined.index)[WATER_STRESS_CODE_FIELD].first().values
        
        # Define stress-to-score mapping (inverted: low stress = high score)
        stress_score_map = {
            -1: 0.0,   # Arid
            0: 1.0,    # Low
            1: 0.75,   # Low-medium
            2: 0.5,    # Medium-high
            3: 0.25,   # High
            4: 0.0,    # Extremely high
        }
        
        score_array = np.array(
            [stress_score_map.get(int(code) if pd.notna(code) else -999, 0.0)
             for code in stress_codes],
            dtype=float
        )
        
        # Map back to original grid order
        score_series = pd.Series(score_array, index=joined.groupby(joined.index).first().index)
        score = score_series.reindex(grid_metric.index, fill_value=0.0).values
        
        return score
    except Exception as e:
        st.warning(f"Error computing water stress score: {e}")
        return score


def score_to_rgba(score: float, min_alpha: int = 50, max_alpha: int = 240) -> list:
    """
    Convert a suitability score [0, 1] to RGBA color [R, G, B, A].
    
    Uses a white→green ramp:
    - score=0.0 → white, fully transparent [255, 255, 255, 0]
    - score~0.5 → light green, moderate alpha
    - score=1.0 → dark forest green, opaque [34, 139, 34, 240]
    
    Args:
        score: Float in [0, 1]
        min_alpha: Alpha at score=0 (default 0 for transparency)
        max_alpha: Alpha at score=1 (default 240 for visibility)
    
    Returns:
        [R, G, B, A] list for pydeck PolygonLayer
    """
    if not (0 <= score <= 1):
        score = np.clip(score, 0, 1)
    
    # Linear interpolation from white to forest green
    # White [255, 255, 255] → Forest Green [34, 139, 34]
    r = int(255 - score * (255 - 34))
    g = int(255 - score * (255 - 139))
    b = int(255 - score * (255 - 34))
    a = int(min_alpha + score * (max_alpha - min_alpha))
    
    return [r, g, b, a]


def render_heatmap(ctx: dict, selected_layers: list, dc_gdf: gpd.GeoDataFrame, projection: str, gravity: int, site_size: int):
    """
    Render crisp, clipped severity-weighted AND-logic suitability heatmap using pydeck PolygonLayer.
    
    Pipeline:
    1. Build 5km grid in EPSG:5070 (metric), convert to EPSG:4326 for rendering.
    2. Compute US boundary mask (area-based, 95% overlap threshold) using dissolve renewables outline.
    3. Compute allowed area mask (area-based, 95% overlap) using inverse-urban polygon.
    4. For selected metric layers, compute severity scores via spatial join:
       - Renewables: grid centroid → renewable_pct polygon → binned to [0.25, 0.5, 0.75, 1.0]
       - Water Stress: grid centroid → stress code polygon → inverted to [0.0, 0.25, 0.5, 0.75, 1.0]
    5. Combine scores with AND logic (min) across selected layers.
    6. Apply masks: final_score *= US_MASK * ALLOWED_MASK.
    7. Convert scores to RGBA using white→green ramp.
    8. Render as crisp PolygonLayer (non-transparent) with NO basemap.
    9. Add US outline stroke layer (no fill).
    10. Add data center point overlay (teal dots).
    
    Args:
        ctx: Context dict with loaded layers (water_us, renewables_us, urban_us)
        selected_layers: List of layer names ("Water Stress", "Renewables", "Inverse Urban Zones")
        dc_gdf: GeoDataFrame of data centers
        projection, gravity, site_size: For context/reference (not used in scoring)
    """
    # === Create/cache 5km grid in metric and geographic CRS ===
    grid_metric, grid_geo, centroids_metric, centroids_geo = create_grid_5km()
    grid_size = len(grid_metric)
    
    # === Compute hard constraint masks (area-based with 95% overlap threshold) ===
    # US boundary: strict clipping to renewables-dissolve outline
    us_mask = compute_us_mask_area_based(grid_metric, ctx["renewables_us"], overlap_threshold=0.95)
    
    # Allowed area: strict filtering of urban zones using inverse polygon
    inverse_urban_geom = ctx["urban_us"].geometry.iloc[0] if len(ctx["urban_us"]) > 0 else None
    # Convert to metric CRS if available
    if inverse_urban_geom is not None:
        from shapely.geometry import shape
        inverse_urban_metric = gpd.GeoSeries([inverse_urban_geom], crs=GEO_CRS).to_crs(METRIC_CRS).iloc[0]
    else:
        inverse_urban_metric = None
    
    allowed_mask = compute_allowed_mask_area_based(grid_metric, inverse_urban_metric, overlap_threshold=0.95)
    
    # === Compute severity scores for selected metric layers ===
    final_score = np.ones(grid_size, dtype=float)  # Start with 1.0 for AND logic (min)
    metric_layer_count = 0
    
    if "Renewables" in selected_layers:
        renewables_score = compute_renewables_score(centroids_metric, ctx["renewables_us"], grid_metric)
        final_score = np.minimum(final_score, renewables_score)
        metric_layer_count += 1
    
    if "Water Stress" in selected_layers:
        water_score = compute_water_stress_score(centroids_metric, ctx["water_us"], grid_metric)
        final_score = np.minimum(final_score, water_score)
        metric_layer_count += 1
    
    # If no metric layers selected, warn and return
    if metric_layer_count == 0:
        st.warning("No metric layers selected. Please choose 'Renewables' or 'Water Stress'.")
        return None
    
    # === Apply hard masks ===
    # US boundary: zero all cells outside US shape
    # Allowed area: zero all cells in urban areas
    final_score *= us_mask.astype(float)
    final_score *= allowed_mask.astype(float)
    
    # === Filter to cells with non-zero score ===
    valid = final_score > 0
    
    if valid.sum() == 0:
        st.warning("No cells meet the suitability criteria. Check your layer selections.")
        return None
    
    # === Build tile rendering table (polygons with colors) ===
    # Only include cells with score > 0 for rendering
    tile_features = []
    for idx in np.where(valid)[0]:
        rgba = score_to_rgba(final_score[idx])
        geom = grid_geo.geometry.iloc[idx]
        coords = list(geom.exterior.coords)
        
        feature = {
            "type": "Feature",
            "properties": {
                "score": float(final_score[idx]),
                "fill_color": rgba,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
        }
        tile_features.append(feature)
    
    if len(tile_features) == 0:
        st.warning("No valid tiles to render.")
        return None
    
    tiles_geojson = {
        "type": "FeatureCollection",
        "features": tile_features,
    }
    
    # === Create polygon tiles layer (RGBA colors, crisp rendering) ===
    tiles_layer = pdk.Layer(
        "PolygonLayer",
        data=tiles_geojson,
        filled=True,
        stroked=False,
        pickable=False,
        get_fill_color="properties.fill_color",
    )
    
    # === Create US outline stroke layer (no fill) ===
    # Use dissolved renewables boundary as the authoritative US shape
    try:
        us_boundary = ctx["renewables_us"].dissolve().geometry.iloc[0]
        us_outline_gdf = gpd.GeoDataFrame(
            geometry=[us_boundary],
            crs=GEO_CRS
        )
        us_outline_geojson = pdk.data.to_geojson(us_outline_gdf)
        
        us_outline_layer = pdk.Layer(
            "PolygonLayer",
            data=us_outline_geojson,
            filled=False,
            stroked=True,
            line_width_min_pixels=1.5,
            get_line_color=[31, 41, 55, 200],  # Dark gray
            pickable=False,
        )
    except Exception as e:
        st.warning(f"Could not render US outline: {e}")
        us_outline_layer = None
    
    # === Prepare data center overlay (teal dots, non-interactive) ===
    dc_us = dc_gdf.cx[US_MINX:US_MAXX, US_MINY:US_MAXY].copy()
    if len(dc_us) > 0:
        dc_points = dc_us.copy()
        dc_points["geometry"] = dc_points.geometry.representative_point()
        # Duplicate each point to ensure visibility on top of tiles
        dc_geojson = pdk.data.to_geojson(dc_points)
        
        dc_layer = pdk.Layer(
            "ScatterplotLayer",
            data=dc_geojson,
            get_position="coordinates",
            get_color=[0, 122, 110],  # Teal
            get_radius=150,
            pickable=False,
        )
    else:
        dc_layer = None
    
    # === Assemble layers (order matters for z-index) ===
    layers = [tiles_layer]
    if us_outline_layer is not None:
        layers.append(us_outline_layer)
    if dc_layer is not None:
        layers.append(dc_layer)
    
    # === Set viewport centered on US ===
    view_state = pdk.ViewState(
        latitude=(US_MINY + US_MAXY) / 2,
        longitude=(US_MINX + US_MAXX) / 2,
        zoom=3.5,
        pitch=0,
    )
    
    # === Create Deck with NO basemap ===
    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=None,  # NO Mapbox tiles; clean slate
        tooltip=None,
    )
    
    return r


@st.cache_resource(show_spinner=False)
def load_context() -> dict:
    # Check if parquet files exist (fast path)
    use_parquet = all([WATER_STRESS_PARQUET.exists(), RENEWABLES_PARQUET.exists(), URBAN_ZONES_PARQUET.exists()])
    
    if use_parquet:
        # Load from cached Parquet files
        water_us = gpd.read_parquet(WATER_STRESS_PARQUET)
        us_outline = water_us.dissolve().boundary
        
        renewables_us = gpd.read_parquet(RENEWABLES_PARQUET)
        us_outline_renewables = renewables_us.dissolve().boundary
        
        urban_non_zones_gdf = gpd.read_parquet(URBAN_ZONES_PARQUET)
        us_outline_urban = urban_non_zones_gdf.dissolve().boundary
    else:
        # Fallback: read from original files and simplify (slow, one-time only)
        for path in [AQUEDUCT_CSV, AQUEDUCT_GDB, PROJECTED_DC_DIR, EGRID_XLSX, EGRID_SUBREGION_KMZ, URBAN_POLYGONS_SHP]:
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
        water_gdf = water_gdf.set_crs(GEO_CRS) if water_gdf.crs is None else water_gdf.to_crs(GEO_CRS)
        water_us = water_gdf.cx[US_MINX:US_MAXX, US_MINY:US_MAXY].copy()
        
        # Simplify in metric CRS (100m tolerance), then convert back
        water_us = water_us.to_crs(METRIC_CRS)
        water_us["geometry"] = water_us.geometry.simplify(100)  # 100 meters
        water_us = water_us.to_crs(GEO_CRS)
        us_outline = water_us.dissolve().boundary

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
        
        # Simplify in metric CRS (100m tolerance), then convert back
        renewables_us = renewables_us.to_crs(METRIC_CRS)
        renewables_us["geometry"] = renewables_us.geometry.simplify(100)  # 100 meters
        renewables_us = renewables_us.to_crs(GEO_CRS)
        us_outline_renewables = renewables_us.dissolve().boundary
        
        urban_gdf = gpd.read_file(URBAN_POLYGONS_SHP)[["geometry"]].copy()
        urban_gdf = urban_gdf.set_crs(GEO_CRS) if urban_gdf.crs is None else urban_gdf.to_crs(GEO_CRS)
        
        # Buffer and simplify in metric CRS
        urban_gdf = urban_gdf.to_crs(METRIC_CRS)
        urban_gdf["geometry"] = urban_gdf.geometry.buffer(1609.34)  # 1 mile in meters
        urban_gdf["geometry"] = urban_gdf.geometry.simplify(100)    # 100 meters
        urban_gdf = urban_gdf.to_crs(GEO_CRS)
        
        urban_us = urban_gdf.cx[US_MINX:US_MAXX, US_MINY:US_MAXY].copy()
        
        # Create inverse: everything in the US EXCEPT urban zones
        us_box = box(US_MINX, US_MINY, US_MAXX, US_MAXY)
        urban_union = urban_us.unary_union if len(urban_us) > 0 else None
        if urban_union is not None:
            urban_inverse_geom = us_box.difference(urban_union)
        else:
            urban_inverse_geom = us_box
        urban_non_zones_gdf = gpd.GeoDataFrame(geometry=[urban_inverse_geom], crs=GEO_CRS)
        us_outline_urban = urban_non_zones_gdf.dissolve().boundary
    
    # Load data center projections (not cached as files)
    projection_dirs = sorted([d for d in PROJECTED_DC_DIR.iterdir() if d.is_dir()])
    dc_by_projection_and_gravity = {}
    for proj_dir in projection_dirs:
        projection = proj_dir.name
        geojson_files = sorted(proj_dir.glob(f"{projection}_*_market_gravity.geojson"))
        by_gravity = {}
        for fp in geojson_files:
            gravity = int(fp.stem.split("_")[-3])
            gdf = gpd.read_file(fp)
            gdf = gdf.set_crs(GEO_CRS) if gdf.crs is None else gdf.to_crs(GEO_CRS)
            by_gravity[gravity] = gdf
        if by_gravity:
            dc_by_projection_and_gravity[projection] = by_gravity

    return {
        "water_us": water_us,
        "us_outline": us_outline,
        "renewables_us": renewables_us,
        "us_outline_renewables": us_outline_renewables,
        "urban_us": urban_non_zones_gdf,
        "us_outline_urban": us_outline_urban,
        "dc_by_projection_and_gravity": dc_by_projection_and_gravity,
        "projection_values": sorted(dc_by_projection_and_gravity.keys()),
        "gravity_values": sorted({g for by_g in dc_by_projection_and_gravity.values() for g in by_g.keys()}),
    }


def classify_sites_background(background: str, sites: gpd.GeoDataFrame, ctx: dict) -> tuple[gpd.GeoDataFrame, list, str]:
    if background == "Inverse Urban Zones":
        sites_copy = sites.copy()
        sites_copy["site_group"] = "Non-urban area"
        return sites_copy, [], ""
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

    if background == "Inverse Urban Zones":
        ax.set_facecolor("#f5f3f0")
        urban_us = ctx["urban_us"]
        urban_us.plot(ax=ax, color=URBAN_ZONES_COLOR, linewidth=0, alpha=URBAN_ZONES_ALPHA, zorder=2)
        if show_boundaries:
            urban_us.boundary.plot(ax=ax, color="#9f7aea", linewidth=0.3, alpha=0.6, zorder=3)
        ctx["us_outline_urban"].plot(ax=ax, color="#1f2937", linewidth=1.2, alpha=0.95, zorder=6)
    elif background in {"Water stress", "Hybrid"}:
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


@st.cache_data
def get_cached_figure(background: str, projection: str, gravity: int, site_size: int, show_boundaries: bool, ctx_id: int):
    """Cached wrapper for make_figure. Uses ctx_id to identify the context instead of hashing the dict."""
    ctx = st.session_state.get("_ctx_cache")
    if ctx is None:
        raise RuntimeError("Context not available in session state")
    return make_figure(background, projection, gravity, site_size, show_boundaries, ctx)


def main():
    st.set_page_config(page_title="Data Center Siting Explorer", layout="wide")
    st.title("Data Center Siting Explorer")
    st.caption("Toggle between water stress, renewable energy, and urban population backgrounds to compare projected data center siting outputs.")

    try:
        ctx = load_context()
        st.session_state["_ctx_cache"] = ctx
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        return

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        view_mode = st.radio("View Mode", ["Single Background", "Multi-Layer Heatmap"], index=0)
        
        projection_default = "high_growth" if "high_growth" in ctx["projection_values"] else ctx["projection_values"][0]
        projection = st.selectbox("Projection", ctx["projection_values"], index=ctx["projection_values"].index(projection_default))
        gravity_default = 25 if 25 in ctx["gravity_values"] else ctx["gravity_values"][0]
        gravity = st.select_slider("Market gravity", options=ctx["gravity_values"], value=gravity_default)
        site_size = st.slider("Site size", min_value=12, max_value=80, value=30, step=2)
        
        if view_mode == "Single Background":
            background = st.radio("Background", ["Water stress", "Renewables %", "Inverse Urban Zones", "Hybrid"], index=0)
            show_boundaries = st.checkbox("Show boundaries", value=False)
        else:
            st.subheader("Severity-Weighted Heatmap")
            st.caption("Green intensity shows combined severity of selected layers, constrained to US outline and non-urban areas.")
            layer_options = ["Water Stress", "Renewables", "Inverse Urban Zones"]
            selected_layers = st.multiselect(
                "Metric Layers",
                layer_options,
                default=["Water Stress", "Renewables"],
                help="Water Stress: low stress→more green. Renewables: high %→more green. Inverse Urban Zones: hard filter (masks out urban areas)."
            )
    
    # Main content area (outside sidebar context)
    if view_mode == "Single Background":
        fig = get_cached_figure(background, projection, gravity, site_size, show_boundaries, id(ctx))
        st.pyplot(fig, clear_figure=True)
    else:
        if not selected_layers:
            st.warning("Please select at least one layer.")
        else:
            # Get data centers for the selected projection and gravity
            dc_gdf = ctx["dc_by_projection_and_gravity"][projection][gravity].copy()
            
            try:
                deck = render_heatmap(ctx, selected_layers, dc_gdf, projection, gravity, site_size)
                if deck is not None:
                    st.pydeck_chart(deck)
            except Exception as e:
                st.error(f"Error rendering heatmap: {e}")


if __name__ == "__main__":
    main()