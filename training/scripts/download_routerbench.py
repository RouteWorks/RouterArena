from huggingface_hub import hf_hub_download
import pickle
import os

os.makedirs("./data/routerbench", exist_ok=True)

path = hf_hub_download(
    repo_id="withmartian/routerbench",
    filename="routerbench_0shot.pkl",
    repo_type="dataset",
    local_dir="./data/routerbench",
)
print(f"Downloaded to {path}")

with open("./data/routerbench/routerbench_0shot.pkl", "rb") as f:
    data = pickle.load(f)

print(f"Type: {type(data)}")
if hasattr(data, "columns"):
    print(f"Columns: {list(data.columns)}")
    print(f"Shape: {data.shape}")
    print(data.head(2))
