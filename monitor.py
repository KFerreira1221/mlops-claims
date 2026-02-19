import json
import pandas as pd
import numpy as np

meta = json.load(open("models/metadata.json"))
logs = [json.loads(x) for x in open("logs/predictions.jsonl")]

df = pd.DataFrame([r["features"] for r in logs])

print("Drift Report")
for feat, stats in meta["baseline_numeric_stats"].items():
    if feat in df.columns:
        cur_mean = np.nanmean(pd.to_numeric(df[feat], errors="coerce"))
        shift = abs(cur_mean - stats["mean"]) / (stats["std"] + 1e-9)
        if shift > 3:
            print(f"{feat} drift detected: {shift:.2f} std devs")
