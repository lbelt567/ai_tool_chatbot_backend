import pandas as pd
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import pickle

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load your cleaned CSV
df = pd.read_csv("all_ai_tools_cleaned.csv")

# Combine useful columns into one text field for semantic search
df["text"] = (
    df["Name"].fillna("") + " — " +
    df["Short Description"].fillna("") + " | " +
    df["Categories"].fillna("") + " | " +
    df["Price Text"].fillna("")
)

texts = df["text"].tolist()

# Get embeddings
def get_embedding(text, retries=3):
    for _ in range(retries):
        try:
            return client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            ).data[0].embedding
        except Exception as e:
            print(f"Retrying after error: {e}")
            time.sleep(2)
    raise Exception("Failed to get embedding after retries.")

print("Generating embeddings...")
embeddings = [get_embedding(t) for t in texts]

# Save FAISS index
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

faiss.write_index(index, "tool_index.faiss")
df.to_csv("tool_metadata.csv", index=False)

# ✅ Save metadata as pickle
tool_metadata = df
with open("tool_metadata.pkl", "wb") as f:
    pickle.dump(tool_metadata, f)

print("✅ Index built and saved!")
