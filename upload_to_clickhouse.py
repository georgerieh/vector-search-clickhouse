import json

import clickhouse_connect
import clip
import numpy as np
import pandas as pd

data = []
with open("/Volumes/T7/photos_from_icloud-out/metadata.jsonl") as f:
    for line in f:
        data.append(json.loads(line))
df1 = pd.DataFrame(
    data,
    columns=[
        "path",
        "file_id",
        "subfolder",
        "date",
        "height",
        "width",
        "location",
        "text",
        "lat",
        "lon",
    ],
)
df1["filename"] = (
        "/Volumes/T7/photos_from_icloud/" + df1["subfolder"] + "/" + df1["file_id"]
)
data = []
with open("/Volumes/T7/photos_from_icloud-out/embeddings_new.jsonl") as f:
    for line in f:
        data.append(json.loads(line).values())


def normalize_vector(vector: list) -> list:
    arr = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()  # avoid divide by zero
    normalized = arr / norm
    return normalized.tolist()


df2 = pd.DataFrame(data, columns=["filename", "facenet_embeddings", "dino_embedding"])
df_combined = pd.merge(df1, df2, how="inner", on="filename")
df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
df_combined["paths"] = "/Volumes/T7/photos_from_icloud/" + df_combined["subfolder"]
df_combined["path"] = df_combined["paths"] + "/" + df_combined["file_id"]
df = df_combined
df = df.loc[:, ~df.columns.duplicated()].copy()
df = df[[i for i in df.columns if i not in ["paths", "file_id"]]]
df["dino_embedding"] = df["dino_embedding"].apply(
    lambda x: list(x) if isinstance(x, (list, np.ndarray)) else [0.0] * 768
)
df["facenet_embeddings"] = df["facenet_embeddings"].apply(
    lambda x: list(x) if isinstance(x, (list, np.ndarray)) else []
)
df["location"] = df["location"].apply(
    lambda x: json.dumps(x) if isinstance(x, dict) else ""
)  # +
df["text"] = df["text"].fillna("").astype(str)  # +
df["height"] = df["height"].fillna(0).astype("UInt32")  # +
df["width"] = df["width"].fillna(0).astype("UInt32")  # +
df["lat"] = df.get("lat", 0.0).astype("float32")  # +
df["lon"] = df.get("lon", 0.0).astype("float32")  # +
df["path"] = df["path"].fillna("").astype(str)  # +
df["filename"] = df["filename"].fillna("").astype(str)  # +
df["subfolder"] = df["subfolder"].fillna("").astype(str)
df["date"] = df["date"].fillna("").astype(str)
df = df.explode("facenet_embeddings")
df["facenet_embedding"] = df["facenet_embeddings"].apply(
    lambda x: x.get("embedding", []) if isinstance(x, dict) else [0.0] * 512
)
df.drop("facenet_embeddings", inplace=True, axis=1)
client = clickhouse_connect.get_client(
    host="localhost", port=8123, username="default", password=""
)
client.command("""
               CREATE TABLE IF NOT EXISTS photos_db
               (
    filename String,
    subfolder Nullable(String),
    date Nullable(String),
    height Nullable(UInt32),
    width Nullable(UInt32),
    location          Nullable(String),
    text              Nullable(String),
    dino_embedding    Array(Float32),
    facenet_embedding Array(Float32),
    path Nullable(String),
    lat Nullable(Float32),
    lon Nullable(Float32)
                   ) ENGINE = ReplacingMergeTree
               (
               )
ORDER BY filename;
               """)

client.insert_df("photos_db", df)
count = df.groupby("filename")
print(f"Data uploaded successfully! - {len(df)} photos inserted.")
