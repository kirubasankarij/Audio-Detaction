import pandas as pd

df = pd.read_csv("data/ESC50/meta/esc50.csv")
out = df[["target", "category"]].drop_duplicates().sort_values("target")
print(out.head(60))
