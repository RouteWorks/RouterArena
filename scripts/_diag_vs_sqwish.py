# SPDX-FileCopyrightText: 2026 Chuzom (github.com/ypollak2/chuzom)
# SPDX-License-Identifier: MIT
"""Diagnostic: where Sqwish beats us on the full dataset."""

import json
from collections import Counter, defaultdict
from typing import Any

ours = json.load(open("router_inference/predictions/llm-router.json"))
sqwish = json.load(open("router_inference/predictions/sqwish-router.json"))

sqw_by_idx = {d["global index"]: d for d in sqwish if not d.get("for_optimality")}

miss: defaultdict[str, dict[str, Any]] = defaultdict(
    lambda: {"count": 0, "sqwish_picks": Counter(), "our_picks": Counter()}
)
sqwish_wins = 0
both_right = 0
both_wrong = 0
we_win = 0
total = 0

for d in ours:
    if d.get("for_optimality"):
        continue
    idx = d["global index"]
    if idx not in sqw_by_idx:
        continue
    total += 1
    our_acc = (d.get("accuracy") or 0) >= 0.5
    sqw_acc = (sqw_by_idx[idx].get("accuracy") or 0) >= 0.5

    if not our_acc and sqw_acc:
        sqwish_wins += 1
        ds = idx.rsplit("_", 1)[0]
        miss[ds]["count"] += 1
        miss[ds]["sqwish_picks"][sqw_by_idx[idx]["prediction"]] += 1
        miss[ds]["our_picks"][d["prediction"]] += 1
    elif our_acc and sqw_acc:
        both_right += 1
    elif not our_acc and not sqw_acc:
        both_wrong += 1
    elif our_acc and not sqw_acc:
        we_win += 1

print(
    f"Total: {total} | Both right: {both_right} | Both wrong: {both_wrong} | "
    f"Sqwish wins: {sqwish_wins} | We win: {we_win}"
)
print(
    f"Net delta: Sqwish ahead by {sqwish_wins - we_win} prompts "
    f"({(sqwish_wins - we_win) / total * 100:.2f}%)"
)
print()
print("Top 12 datasets where Sqwish wins but we lose:")
rows = sorted(miss.items(), key=lambda kv: -kv[1]["count"])[:12]
for ds, info in rows:
    sqw_top = ", ".join(
        f"{m.split('/')[-1][:18]}={c}" for m, c in info["sqwish_picks"].most_common(2)
    )
    our_top = ", ".join(
        f"{m.split('/')[-1][:18]}={c}" for m, c in info["our_picks"].most_common(2)
    )
    print(f"  {ds:32s} {info['count']:3d}")
    print(f"    sqwish picks: {sqw_top}")
    print(f"    we pick:      {our_top}")
