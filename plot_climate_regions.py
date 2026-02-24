import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

climdiv = gpd.read_file('/Users/jaydenudall/Desktop/datathon data/data/CONUS_CLIMATE_DIVISIONS/GIS.OFFICIAL_CLIM_DIVISIONS.shp')
climdiv = climdiv.to_crs("EPSG:4326")

climate_df = pd.read_csv('/Users/jaydenudall/Desktop/datathon data/data/climate_data.csv')

climate_avg = (
    climate_df
    .groupby("CLIMDIV", as_index=False)
    .agg({
        "wetbulb": "mean",
        "wind": "mean",
        "precip": "mean"
    })
)

climdiv["CLIMDIV"] = climdiv["CLIMDIV"].astype(int)
climate_avg["CLIMDIV"] = climate_avg["CLIMDIV"].astype(int)

climdiv = climdiv.merge(climate_avg, on="CLIMDIV", how="left")

# calculate predicted WUE without renewable energy share
climdiv["predicted_WUE"] = (
    1.1257
    + 0.0079 * climdiv["wetbulb"]
    + 0.00004361 * climdiv["wind"]
    - 0.0001 * climdiv["precip"]
)

climdiv = climdiv.to_crs("EPSG:5070")
fig, ax = plt.subplots(figsize=(12,8))

climdiv.plot(
    column="predicted_WUE",
    cmap="viridis_r",
    legend=True,
    edgecolor="black",
    linewidth=0.15,
    ax=ax,
    missing_kwds={"color":"lightgrey"}
)

# colorbar
fig = ax.get_figure()
wue_cbar = fig.axes[-1]
wue_cbar.set_ylabel("Predicted WUE (minus renewable energy share)", fontsize=11)

ax.set_title("Average Predicted Data Center Water Efficiency by U.S. Climate Division", fontsize=14)
ax.axis("off")
plt.tight_layout()
plt.savefig("us_climate_divisions_wue.png", dpi=300)
plt.show()
