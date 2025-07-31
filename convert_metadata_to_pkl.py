import pandas as pd
import pickle

df = pd.read_csv("tool_metadata.csv")
with open("tool_metadata.pkl", "wb") as f:
    pickle.dump(df, f)

print("✅ Converted tool_metadata.csv to tool_metadata.pkl")
