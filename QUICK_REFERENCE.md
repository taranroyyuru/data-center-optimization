# Quick Start & Testing Guide

## File Summary

### New Files
- **`xlsx_utils.py`** - Standalone XLSX parsing utility (no Streamlit imports)

### Modified Files
- **`streamlit_app.py`** - Fixed mask logic, CRS handling, pydeck properties, added caching
- **`convert_to_parquet.py`** - Updated to use metric CRS, imports from xlsx_utils
- **`FIXES_SUMMARY.md`** - Detailed explanation of all bug fixes (this folder)

---

## Running the App

### Option 1: Use Cached Parquet Files (Recommended, Fast)
```bash
streamlit run streamlit_app.py
```
- Requires: `datasets/cache/water_stress.parquet`, `renewables.parquet`, `urban_zones.parquet`
- Creates these on first run if missing (see Option 2)

### Option 2: Generate Cache Files (One-Time, ~5-10 min)
```bash
python convert_to_parquet.py
```
- Reads raw data: Aqueduct GDB, EGRID KMZ, urban SHP
- Simplifies in EPSG:5070 (100m tolerance)
- Saves to `datasets/cache/` as Parquet
- Cache files persist for fast app loads

---

## Testing the Multi-Layer Heatmap

1. **Start the app** (see Running section above)
2. **Toggle to Multi-Layer Heatmap mode:**
   - Sidebar: Select "Multi-Layer Heatmap" radio button
3. **Select layers:**
   - Check "Water Stress" → Heatmap should render with warm colors
   - Check "Renewables" → Adds green regions
   - Check "Inverse Urban Zones" → Adds non-urban areas
   - Toggle multiple together → Heatmap shows overlap (intensity increases)
4. **Verify heatmap properties:**
   - White/light = low overlap (0-1 layers)
   - Light green = medium (1-2 layers)
   - Dark green = high overlap (2+ layers)
   - Data center points overlaid as teal dots

---

## Expected Behavior

### First Run
1. App checks for Parquet files in `datasets/cache/`
2. If missing: Falls back to reading raw files + simplifying (slow, ~30-60s)
3. If present: Loads from Parquet (fast, <2s)

### Layer Toggle Performance
- Toggling layers (checking/unchecking multiselect) should be instant
- Heatmap re-renders from cached masks + sum operation
- No spatial joins on every toggle

### Tooltip & Interaction
- Hover over teal data center points → Shows "Data Center"
- Zoom/pan map with pydeck controls
- Color gradient smooth white→green across US

---

## Debugging Tips

### Issue: Heatmap doesn't render
- Check browser console for JavaScript errors
- Verify `heatmap_points` DataFrame has columns: `lon`, `lat`, `weight`
- Ensure at least one layer is selected
- Check that weight > 0 (some grid cells may have zero overlap)

### Issue: Mask dimensions error
- The fix: `groupby()` aggregation ensures mask length = `len(grid)`
- If you see "ValueError: shape mismatch" → The mask logic has a bug
- Should not occur with the fix applied

### Issue: Slow heatmap computation
- First layer selection is slow (computes masks)
- Subsequent togacles fast (cached)
- If still slow, check:
  - CPU usage (spatial index may need optimization for large GDFs)
  - Grid cell size (increase from 0.15° if too many cells)

### Issue: Parquet files not created
- Ensure raw data files exist:
  - `datasets/Aqueduct40_waterrisk_download_Y2023M07D05/`
  - `grid data/egrid2023_subregions.kmz`
  - `urban population polygons/tl_2025_us_uac20.shp`
- Run: `python convert_to_parquet.py`
- Check console for error messages

---

## Key Constants

### CRS Definitions (streamlit_app.py)
```python
GEO_CRS = "EPSG:4326"     # Geographic (lon/lat)
METRIC_CRS = "EPSG:5070"  # Projected (meters, US-focused)
```

### Grid Parameters
```python
cell_size_degrees = 0.15  # ~15 km cells covering US
US_MINX, US_MAXX = -125.0, -66.5  # longitude bounds
US_MINY, US_MAXY = 24.0, 49.8     # latitude bounds
```

### Geometry Simplification
```python
simplify_tolerance_meters = 100  # In metric CRS before converting back
```

### Heatmap Color Range (RGB)
```python
[255, 255, 255, 0]      # White, transparent (0 overlap)
[220, 245, 220, 200]    # Very light green
[144, 238, 144, 220]    # Light green
[50, 205, 50, 230]      # Lime green
[34, 139, 34, 240]      # Forest green (max overlap)
```

---

## Layer Names (Must Match)

**In UI multiselect:**
- "Water Stress"
- "Renewables"
- "Inverse Urban Zones"

**In Single Background radio:**
- "Water stress" (lowercase 's', different from multiselect)
- "Renewables %" (with percent sign)
- "Inverse Urban Zones"
- "Hybrid"

---

## Architecture Overview

```
User selects layers → render_heatmap()
  ↓
create_cached_grid() → Grid(n_cells), lons[], lats[]
  ↓
For each layer:
  get_layer_mask() → spatial join + groupby → bool[n_cells]
  ↓
Sum masks → weight[n_cells]  (overlap count per cell)
  ↓
Filter weight > 0 → heatmap_points DataFrame(lon, lat, weight)
  ↓
pdk.Deck([HeatmapLayer(data=heatmap_points), ScatterplotLayer(data=dc_points)])
```

---

## Future Optimization Ideas

1. **Mask persistence:** Save `get_cached_layer_mask()` results to disk as `.npy` files
2. **Grid resolution control:** Add slider to adjust `cell_size_degrees` (0.08 to 0.25)
3. **Layer weighting:** Allow user to weight layers differently in overlap sum
4. **Export heatmap:** Save rendered heatmap as GeoTIFF or PNG
5. **Summary stats:** Display total cells, average overlap, coverage %

