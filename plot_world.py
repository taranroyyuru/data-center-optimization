import requests
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import shape
import pandas as pd
import cmocean as cmo

plants_df = pd.read_csv("data/usa_power_plant_data.csv")
electricity_df = pd.read_csv('data/electricity_data.csv')
conus_gdf = gpd.read_file('geojson_files/conus_map.geojson')
climdiv_gdf = gpd.read_file('data/CONUS_CLIMATE_DIVISIONS/GIS.OFFICIAL_CLIM_DIVISIONS.shp')
climate_df = pd.read_csv('data/climate_data.csv')


fuel_colors = {
    "Solar": "#FDB813",      # bright yellow
    "Wind": "#56B4E9",       # sky blue
    "Hydro": "#0072B2",      # deep blue
    "Geothermal": "#E41A1C", # red
    "Biomass": "#2E8B57",    # green
    "Coal": "#4D4D4D",       # dark gray
    "Gas": "#E69F00",        # orange
    "Oil": "#8B4513",        # brown
    "Nuclear": "#CC79A7"     # magenta
}


renewables = ["Solar", "Wind", "Hydro", "Geothermal", "Biomass"]
plants_gdf = plants_df[plants_df["primary_fuel"].isin(renewables)]

plants_gdf = gpd.GeoDataFrame(
    plants_df,
    geometry=gpd.points_from_xy(plants_df.longitude, plants_df.latitude),
    crs="EPSG:4326"
)
plants_gdf["size"] = plants_gdf["capacity_mw"] / 80


map_gdf = conus_gdf.merge(electricity_df, on="zoneName", how="left")

map_gdf = map_gdf[map_gdf['sourceYear'] == 2024]

climdiv = climdiv_gdf.to_crs("EPSG:5070")
conus_gdf = conus_gdf.to_crs("EPSG:5070")
map_gdf = map_gdf.to_crs("EPSG:5070")
plants_gdf = plants_gdf.to_crs(map_gdf.crs)
plants_conus = gpd.sjoin(
    plants_gdf,
    conus_gdf[["geometry"]],
    predicate="within",
    how="inner"
)

map_gdf["geometry"] = map_gdf.buffer(0)
climdiv["geometry"] = climdiv.buffer(0)


climdiv = climdiv.merge(climate_df, on="CLIMDIV")

print(climdiv.head())


climdiv_pts = climdiv.copy()

climdiv_pts["geometry"] = climdiv.geometry.representative_point()

climdiv_pts = climdiv_pts.to_crs(map_gdf.crs)

climdiv_pts = gpd.sjoin(
    climdiv_pts,
    map_gdf[["zoneName", "renewableEnergyPercentage", "geometry"]],
    predicate="within",
    how="left"
)
climdiv_pts["predicted_WUE"] = (
    1.1257
    + 0.0079 * climdiv_pts["wetbulb"]
    + 0.00004361 * climdiv_pts["wind"]
    - 0.0001 * climdiv_pts["precip"]
    + 0.0071 * climdiv_pts["renewableEnergyPercentage"]
)
# normalize
wue_min = climdiv_pts["predicted_WUE"].min()
wue_max = climdiv_pts["predicted_WUE"].max()

worst = climdiv_pts["predicted_WUE"].max()
climdiv_pts["improvement"] = worst - climdiv_pts["predicted_WUE"]

impr_min = climdiv_pts["improvement"].min()
impr_max = climdiv_pts["improvement"].max()

climdiv_pts["score"] = (climdiv_pts["improvement"] - impr_min) / (impr_max - impr_min)
climdiv_pts["score"] = climdiv_pts["score"] ** 2.5

fig, ax = plt.subplots(figsize=(13,8))

map_gdf.plot(
    column="carbonIntensityDirect",
    cmap="Purples",
    edgecolor="black",
    linewidth=0.4,
    legend=True,
    ax=ax,
    missing_kwds={"color":"lightgrey"}
)

"""
climdiv_pts.plot(
    ax=ax,
    column="score",
    cmap="bone",
    markersize=climdiv_pts["score"] * 20,
    legend=True,
    #edgecolor="black",
    #linewidth=0.3,
    #alpha=0.85,
    #zorder=3
)
"""

plants_conus.plot(
    ax=ax,
    markersize=plants_gdf["size"] * 0.80,
    color="black",
    alpha=0.7
)

fig = ax.get_figure()

# colorbar 1 = carbon map
carbon_cbar = fig.axes[-1]
carbon_cbar.set_ylabel("Grid carbon intensity (gCO₂ / kWh)", fontsize=11)

"""
# colorbar 2 = WUE points
wue_cbar = fig.axes[-1]
wue_cbar.set_ylabel("Predicted WUE", fontsize=11)
"""

ax.set_title(
    "U.S. Grid Carbon Intensity with Predicted Data-Center Water Efficiency Locations",
    fontsize=14,
    pad=12
)

ax.axis("off")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()


