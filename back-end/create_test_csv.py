import numpy as np
import pandas as pd

# Step 1: set up date range
n = 365
dates = pd.date_range(start="2020-01-01", periods=n, freq="D")

# Step 2: generate series
flat_values = 100 + np.random.normal(0, 1, n)
ramp_values = np.linspace(95, 105, n)
wave_values = 100 + 5 * np.sin(2 * np.pi * np.arange(n) / 30)

# Step 3: create DataFrames
flat_df = pd.DataFrame({"date": dates, "value": flat_values})
ramp_df = pd.DataFrame({"date": dates, "value": ramp_values})
wave_df = pd.DataFrame({"date": dates, "value": wave_values})

# Step 4: write to CSV
flat_df.to_csv("flat.csv", index=False)
ramp_df.to_csv("ramp.csv", index=False)
wave_df.to_csv("wave.csv", index=False)
