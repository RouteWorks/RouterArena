# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

from huggingface_hub import snapshot_download
import os

os.makedirs("./data/r2bench", exist_ok=True)

path = snapshot_download(
    repo_id="JiaqiXue/R2-Bench-RouterArena",
    repo_type="dataset",
    local_dir="./data/r2bench",
)
print(f"Downloaded to {path}")

for root, dirs, files in os.walk("./data/r2bench/data"):
    for f in files[:3]:
        print(os.path.join(root, f))
    break
