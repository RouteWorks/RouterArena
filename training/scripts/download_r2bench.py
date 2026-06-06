from datasets import load_dataset


ds = load_dataset("UCF-ML-Research/R2-Bench", trust_remote_code=True)
ds.save_to_disk("./data/r2bench")
print(f"Downloaded R2-Bench: {ds}.")
