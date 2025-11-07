#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""Extract metrics from evaluation output and save to JSON file."""

import json
import re
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: extract_metrics.py <output_file>")
        sys.exit(1)

    output_file = sys.argv[1]

    try:
        with open(output_file, "r") as f:
            content = f.read()

        # Find the Metrics JSON block
        match = re.search(r"Metrics:\s*(\{.*?\})", content, re.DOTALL)
        if match:
            metrics_json = match.group(1)
            metrics = json.loads(metrics_json)

            # Write metrics to file
            with open("metrics.json", "w") as mf:
                json.dump(metrics, mf)

            # Output key metrics as step outputs
            print(f"accuracy={metrics['accuracy']}")
            print(f"arena_score={metrics['arena_score']}")
            print(f"total_cost={metrics['total_cost']}")
            print(f"num_queries={metrics['num_queries']}")
        else:
            print("No metrics found in output", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error extracting metrics: {e}", file=sys.stderr)
        sys.exit(1)
