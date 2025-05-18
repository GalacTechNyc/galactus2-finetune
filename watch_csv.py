import time, json, pandas as pd, pathlib
CSV = pathlib.Path("galactus_data.csv")
JSON = pathlib.Path("galactus_dataset.json")
while True:
    if CSV.stat().st_mtime > JSON.stat().st_mtime:
        df = pd.read_csv(CSV).fillna("")
        df.to_json(JSON, orient="records", indent=2, force_ascii=False)
        print("🔄  Dataset regenerated")
    time.sleep(2)
