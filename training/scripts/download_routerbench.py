from datasets import load_dataset

ds = load_dataset("withmartian/routerbench")
ds.save_to_disk("./data/routerbench")
print(f"Downloaded {len(ds['train'])} training records.")