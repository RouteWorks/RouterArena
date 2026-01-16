# Parallel Inference Quick Start Guide

## Overview

This directory now supports parallel inference for processing models efficiently. The system processes models sequentially but uses multiple workers to parallelize queries within each model.

## Quick Start

### 1. Process All Models (Recommended)

Process all 26 models from `model_cost/model_cost.json` with 16 workers per model:

```bash
cd <path_to_your_project_root>
uv run python llm_inference/batch_inference.py --num-workers 16
```

**What this does:**
- Loads all model names from `model_cost/model_cost.json`
- Processes each model sequentially
- Uses 16 workers per model to process queries in parallel
- Skips already processed queries automatically
- Saves results to `./cached_results/{model_name}.jsonl`

### 2. Process Specific Models

Process only certain models:

```bash
uv run python llm_inference/batch_inference.py \
    --models gemini-2.0-flash-001 gpt-5-mini claude-sonnet-4-5 \
    --num-workers 16
```

### 3. Single Model Inference

Process a single model with parallel workers:

```bash
# With 16 workers (parallel)
uv run python llm_inference/main.py \
    --model_name gemini-2.0-flash-001 \
    --num-workers 16

# Sequential (backward compatible)
uv run python llm_inference/main.py \
    --model_name gemini-2.0-flash-001
```

## Configuration Options

### batch_inference.py

| Option | Default | Description |
|--------|---------|-------------|
| `--num-workers` | 16 | Number of parallel workers per model |
| `--models` | All | Specific models to process (space-separated) |
| `--cache-dir` | `./cached_results` | Cache directory path |
| `--model-cost-path` | `./model_cost/model_cost.json` | Path to model cost file |
| `--input-file` | `./llm_inference/datasets/router_data.json` | Input data file |

### main.py

| Option | Default | Description |
|--------|---------|-------------|
| `--model_name` | Required | Model name to process |
| `--num-workers` | 1 | Number of parallel workers (1 = sequential) |
| `--run-full` | False | Process full dataset |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  batch_inference.py                                 │
│  ┌───────────────────────────────────────────────┐ │
│  │  Load models from model_cost.json (26 models)│ │
│  │  Load dataset (8400 queries)                  │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│  FOR EACH MODEL (Sequential):                      │
│  ┌───────────────────────────────────────────────┐ │
│  │  Model 1: gemini-3-pro-preview                │ │
│  │  ┌─────────────────────────────────────────┐ │ │
│  │  │  Check cache → 8200 done, 200 remaining │ │ │
│  │  │  Launch 16 workers (Parallel)            │ │ │
│  │  │  ┌──────┐ ┌──────┐       ┌──────┐       │ │ │
│  │  │  │Worker│ │Worker│  ...  │Worker│       │ │ │
│  │  │  │  1   │ │  2   │       │  16  │       │ │ │
│  │  │  │~12 q │ │~12 q │       │~13 q │       │ │ │
│  │  │  └──────┘ └──────┘       └──────┘       │ │ │
│  │  │  Wait for completion                     │ │ │
│  │  │  Save results                            │ │ │
│  │  └─────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────┐ │
│  │  Model 2: gemini-3-flash-preview             │ │
│  │  (Same parallel processing...)                │ │
│  └───────────────────────────────────────────────┘ │
│  ...                                                │
│  ┌───────────────────────────────────────────────┐ │
│  │  Model 26: meta-llama_llama-3.1-405b-instruct│ │
│  │  (Same parallel processing...)                │ │
│  └───────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

## Next Steps

After inference completes:

1. **Run Evaluation**:

   ```bash
   uv run python llm_evaluation/batch_evaluate.py \
       --cached-results-dir ./cached_results \
       --max-workers 16
   ```

2. **Compute Scores**:

```bash
   uv run python router_evaluation/compute_scores.py <router_name>
   ```

## Files Modified/Created

1. ✅ `llm_inference/parallel_inference.py` - Parallel inference manager
2. ✅ `llm_inference/pipeline.py` - Added parallel support
3. ✅ `llm_inference/main.py` - Added --num-workers argument
4. ✅ `llm_inference/batch_inference.py` - Batch processing script
5. ✅ `docs/PARALLEL_INFERENCE_IMPLEMENTATION.md` - Implementation details
6. ✅ `llm_inference/README_PARALLEL.md` - This guide

## Support

For issues or questions:
1. Check logs for error messages
2. Review `docs/PARALLEL_INFERENCE_IMPLEMENTATION.md` for details
3. Verify model names in `model_cost/model_cost.json`
4. Ensure dataset is prepared (run prep_datasets.py)
