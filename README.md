# RouterArena

## Installation

### Prerequisites
- Python 3.11+ (we'll install it automatically if needed)
- Internet connection

### Step 1: Install uv (if you don't have it)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Install RouterArena
```bash
cd RouterArena
uv sync
```

That's it! 🎉 All dependencies are installed automatically.

### Step 3: Set Up API Keys
Use the `.env` file in the project root with your API keys. The step is required if you want to use our pipeline to make LLM inferences.

### Step 4: Download Dataset
Run this command to download dataset from [HF dataset](https://huggingface.co/datasets/louielu02/RouterEvalBenchmark).

```bash
uv run python ./scripts/process_datasets/prep_datasets.py
```

## Usage

### 0. Universal Model Name
API providers may use different names for the same model (e.g., `gpt-4o`, `openai/gpt-4o`). We manage this via `./universal_model_names.py`:
- `universal_names`: canonical model names used in this project
- `mapping`: maps external provider names to our canonical names
- `ModelNameManager.get_universal_name()`: converts external names to canonical names  

### 1. LLM Inference

Run inference on a specific model:

```bash
cd llm_inference
uv run python main.py --model_name gpt-4o
```

Options:
- `--model_name`: Model name (e.g., `gpt-4o`, `claude-3-5-sonnet`, `gemini-2.5-pro`)

**Example**:

```bash
# Run inference with GPT-4o
uv run python main.py --model_name gpt-4o
```

**Input**: Questions from `llm_inference/datasets/router_data.json`, which is a local copy from the [huggingface dataset](https://huggingface.co/datasets/louielu02/RouterEvalBenchmark).

**Output**: Results saved to `cached_results/<universal_model_name>.jsonl`

For convenience, you could initialize inference on all models using `$MAX_CONCURRENT` threads with this command:  

```bash
uv run ./run_inference.sh
```

### 2. LLM Evaluation

Evaluate model responses using automated metrics and judge models.

**Single model evaluation**:

```bash
cd llm_evaluation
uv run python evaluate_models.py <universal_model_name> --cached-results-dir ../cached_results/
```

**Batch evaluation** (all models in `../cached_results/`):

```bash
cd llm_evaluation
uv run python batch_evaluate.py
```

Both methods automatically evaluate the query-answer pairs from the corresponding model files in `../cached_results/`. Evaluation results are appended to the same files as additional fields.

**Input**: Inference results from `cached_results/<universal_model_name>.jsonl`

**Output**: Same file with evaluation metrics added (e.g., correctness scores, cost, metric, etc)

### 3. Router Evaluation

Evaluate routers based on their model selection predictions for given queries.

**Step 1: Create Router Predictions**

Create a prediction file in `./router_inference/predictions/<router_name>.json` with the following structure:

```json
[
  {
    "global index": "ArcMMLU_655",
    "prompt": "Question text here...",
    "prediction": "gpt-4o",
    "confidence": 0.85,
    "all confidence": [0.85, 0.10, 0.05]
  }
]
```

**Step 2: Create Router Config**

Create a config file in `./router_inference/config/<router_name>.json`:

```json
{
  "pipeline_params": {
    "router_name": "your_router",
    "models": {
      "gpt-4o": 2.50,
      "claude-3-5-sonnet": 3.00
    }
  }
}
```

*Note: Model costs represent averaged price per 1k queries (based on 10% dataset sample from `./llm_inference/datasets/router_data_10.json`)*

**Step 3: Run Evaluation**

```bash
uv run python ./router_inference/compare_router_accuracy.py
```

**Output**: Generates `./router_inference/all_router_data.json` with evaluation results.

## Cost Tracking

The framework tracks inference costs using `./model_cost/cost.json`:

```json
{
  "gpt-4o": {
    "input_token_price_per_million": 2.50,
    "output_token_price_per_million": 10.00
  }
}
```

Please use the canonical model name defined in `./universal_model_names.py`. Costs are automatically calculated during evaluation and stored in results.

## Output Formats

### Cached Results Format (`./cached_results/<universal_model_name>.jsonl`)

```json
{
  "global_index": "AIME_0",
  "question": "What is 2+2?",
  "llm_selected": "gpt-4o",
  "generated_answer": "The answer is \\boxed{4}.",
  "token_usage": {
    "input_tokens": 50,
    "output_tokens": 10,
    "total_tokens": 60
  },
  "success": true,
  "provider": "openai",
  "evaluation_result": {
    "extracted_answer": "4",
    "ground_truth": "4",
    "score": 1.0,
    "metric": "math_metric",
    "inference_cost": 0.00015
  }
}
```

## Citation:
If you find our project helpful, please give us a star and cite us by:

```bibtax
@misc{lu2025routerarenaopenplatformcomprehensive,
  title        = {RouterArena: An Open Platform for Comprehensive Comparison of LLM Routers},
  author       = {Yifan Lu and Rixin Liu and Jiayi Yuan and Xingqi Cui and Shenrun Zhang and Hongyi Liu and Jiarong Xing},
  year         = {2025},
  eprint       = {2510.00202},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2510.00202}
}
```
