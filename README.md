# Data Center Optimization: Codebase Guide

This README is a navigation guide for the repository: where data lives, which notebook does what, and where to find each analysis output.

## Repository Layout

- `waterstress.ipynb`
  - Main analysis notebook.
  - Contains map overlays, scoring logic, and CSV exports.
- `exploration.ipynb`
  - Earlier exploratory workflow (state-level carbon/renewables + IM3 merge experiments).
- `streamlit_app.py`
  - Interactive app version of the mapping workflow.
- `datasets/`
  - Source data used by notebooks/app.
- `exports/projected_site_drought_stress/`
  - Generated CSV exports for drought stress by growth scenario.

## Data Folders

- `datasets/im3_projected_data_centers/`
  - Projected site polygons by growth scenario and market gravity.
  - Scenarios:
    - `low_growth`
    - `moderate_growth`
    - `high_growth`
    - `higher_growth`
  - Each scenario has `*_0/25/50/75/100_market_gravity.geojson`.

- `datasets/Aqueduct40_waterrisk_download_Y2023M07D05/`
  - Aqueduct 4.0 water risk data.
  - CSVs:
    - `CVS/Aqueduct40_future_annual_y2023m07d05.csv`
    - `CVS/Aqueduct40_baseline_monthly_y2023m07d05.csv`
    - `CVS/waterstress_us.csv` (US-only subset generated in notebook)
  - Geometry:
    - `GDB/Aq40_Y2023D07M05.gdb/` (tracked partly through Git LFS)

- `datasets/egrid2023_data_rev2 (2).xlsx`
  - eGRID 2023 data (renewables metrics used in workflows).

- `datasets/Data_Centers_Database.xlsx`
  - Additional reference dataset used in supporting analysis.

## Main Workflow: `waterstress.ipynb`

Use this notebook as the primary workflow. Recommended execution order is top-to-bottom.

### What each key section does

- Cell `0` (markdown)
  - Notebook overview.

- Cells `1-3` (setup + loading)
  - Load Aqueduct data and projected data center files.
  - Build `dc_by_projection_and_gravity` for all growth scenarios and gravity settings.

- Cell `4-5` (water-stress overlay)
  - Interactive US map:
    - Water stress polygons
    - Site markers
    - Projection + gravity toggles

- Cell `6` (renewables layer)
  - Build renewables subregion layer from eGRID + subregion geometry.
  - Adds renewables-based map logic.

- Cell `7` (unified background toggle)
  - Interactive map with background switch:
    - Water stress
    - Renewables %
    - Hybrid
  - Site markers styled by local context.

- Cell `8` (sustainability scoring model)
  - Computes:
    - `im3_norm`
    - `water_good`
    - `renew_good`
    - `env_score`
    - `final_index`
  - Includes weight presets:
    - Economic-first
    - Balanced
    - Sustainability-first
  - Includes optional penalty model.
  - **Finding surfaced here**:
    - Ranked site list by `final_index`.
    - `final_index` summary (`min`, `mean`, `max`).

- Cell `9` (polygon + point heatmap)
  - Polygon heatmap by **environmental score** (`env_score` aggregated by region).
  - Point heatmap by **sustainability score** (`final_index`).
  - **Finding surfaced here**:
    - Regional environmental suitability patterns + site-level sustainability variation.

- Cell `11` (drought-stress exports by scenario)
  - Writes:
    - `exports/projected_site_drought_stress/high_growth_projected_sites_drought_stress.csv`
    - `exports/projected_site_drought_stress/moderate_growth_projected_sites_drought_stress.csv`
    - `exports/projected_site_drought_stress/low_growth_projected_sites_drought_stress.csv`
    - `exports/projected_site_drought_stress/higher_growth_projected_sites_drought_stress.csv`
  - **Finding surfaced here**:
    - Per-site drought stress factor (`1-5`) across all scenario+gravity combinations.

- Cell `13` (US-only Aqueduct CSV export)
  - Writes:
    - `datasets/Aqueduct40_waterrisk_download_Y2023M07D05/CVS/waterstress_us.csv`
  - **Finding surfaced here**:
    - US-filtered annual Aqueduct table for smaller/faster downstream use.

## Exploratory Workflow: `exploration.ipynb`

This notebook is exploratory and not the current primary pipeline.

### Main content

- Load and inspect one IM3 GeoJSON scenario.
- Parse eGRID sheets.
- Build state-level carbon metrics.
- Map IM3 regions to state abbreviations.
- Prototype sustainability score:
  - Carbon normalization
  - Renewable normalization
  - Weighted composite score

### Findings in this notebook

- Early validation that state-level carbon intensity and renewables can be merged into siting context.
- Preliminary sustainability scoring ideas that were later formalized in `waterstress.ipynb`.

## App Workflow: `streamlit_app.py`

Use this for interactive exploration without the notebook UI.

### Features

- Background modes:
  - `Water stress`
  - `Renewables %`
  - `Hybrid`
- Controls:
  - Growth projection
  - Market gravity
  - Marker size
  - Boundary toggle
- Site classification legend and map legends on top layer.

### Run

```bash
python3 -m streamlit run streamlit_app.py --server.port 8501
```

## Outputs You Can Trust As “Final”

- Site-level drought exports:
  - `exports/projected_site_drought_stress/*.csv`
- US-only Aqueduct subset:
  - `datasets/Aqueduct40_waterrisk_download_Y2023M07D05/CVS/waterstress_us.csv`
- Sustainability ranking table:
  - Produced interactively in `waterstress.ipynb` Cell `8` (`final_index`).

## Practical “Where do I go for X?” Guide

- “I need the main analysis and maps”:
  - `waterstress.ipynb`
- “I need ranked sites with sustainability-adjusted score”:
  - `waterstress.ipynb` Cell `8`
- “I need region polygons + site score heatmaps”:
  - `waterstress.ipynb` Cell `9`
- “I need downloadable per-site drought stress files”:
  - Run `waterstress.ipynb` Cell `11`
- “I need the app”:
  - `streamlit_app.py`
- “I need older experiment logic”:
  - `exploration.ipynb`

## Notes / Caveats

- Some Aqueduct GDB assets are large and tracked with Git LFS.
- eGRID subregion geometry availability affects renewables overlays.
  - If geometry is missing, notebook scoring falls back to neutral renewables defaults.
- IM3 features already encode some cooling/cost logic; when increasing environmental weighting, consider potential double counting.
