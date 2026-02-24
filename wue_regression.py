import pandas as pd
import statsmodels.formula.api as smf

cols_to_drop = [
    # other WUE definitions
    "WUE_FixedColdWaterDirect(L/KWh)",
    "WUE_Indirect(L/KWh)",

    # redundant weather variables
    "temperature",
    "humidity",

    # raw energy totals (cause multicollinearity)
    "Total renewables - TWh",
    "Total fossil fuels - TWh",
    "Total energy - TWh",
    "Low carbon - TWh",
    "Other - TWh",

    # detailed generation types (already captured in shares)
    "Other renewables (including geothermal and biomass) - TWh",
    "Biofuels consumption - TWh",
    "Solar consumption - TWh",
    "Wind consumption - TWh",
    "Hydro consumption - TWh",
    "Gas consumption - TWh",
    "Coal consumption - TWh",
    "Oil consumption - TWh",
    "Nuclear consumption - TWh"
]

daily_df = pd.read_csv('data/daily_african_df.csv')
daily_df = daily_df.rename(columns={
    "WUE_FixedApproachDirect(L/KWh)": "WUE",
    "Leakages (%)": "Leakages"
})
daily_df["renew_share"] = daily_df["Total renewables - TWh"] / daily_df["Total energy - TWh"]
daily_df["fossil_share"] = daily_df["Total fossil fuels - TWh"] / daily_df["Total energy - TWh"]
daily_df["nuclear_share"] = daily_df["Nuclear consumption - TWh"] / daily_df["Total energy - TWh"]

daily_df = daily_df.drop(columns=cols_to_drop)

formula = """
WUE ~
wetbulb_temperature +
wind_speed +
precipitation +
renew_share
"""

model = smf.ols(formula, data=daily_df).fit()

print(model.summary())