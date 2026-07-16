import pandas as pd
df = pd.read_csv("src/rxn_location/reconnection_stats_2015-09-02_164500_2015-09-02_174500_10m.csv")
print("r_rc exists?", "r_rc" in df.columns)
print("Columns starting with r_rc:")
print([c for c in df.columns if c.startswith("r_rc")])
print("\nSample values for r_rc across methods:")
print(df[["method_used", "r_rc"]].head())
