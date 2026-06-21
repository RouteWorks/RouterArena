# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0
"""
Domain dataset map: RouterArena categories → HuggingFace proxy datasets.

Each entry specifies:
  - hf_path: HuggingFace dataset path
  - hf_name: config name (or None)
  - split: which split to use (never the exact RouterArena test split)
  - sample_n: max examples to draw
  - label: FLASH / DEEPSEEK / QWEN235B
  - format_fn: key in FORMAT_REGISTRY for prompt construction
  - rationale: why this label (from per-dataset accuracy data)

Label derivation (from RouterArena per-dataset model accuracy):
  FLASH     = gemini-3.1-flash-lite achieves ≥0.50 and is cheapest correct
  DEEPSEEK  = flash-lite <0.50, deepseek-v4-flash ≥0.50
  QWEN235B  = only qwen3-235b ≥0.50, or all models <0.50 (route to best)
"""

from typing import Any

DOMAIN_MAP: list[dict[str, Any]] = [
    # ── FLASH: factual MCQ / knowledge ────────────────────────────────────────
    # ArcMMLU → AI2-ARC (different difficulty levels)
    {
        "ra_datasets": ["ArcMMLU"],
        "hf_path": "allenai/ai2_arc",
        "hf_name": "ARC-Easy",
        "split": "train",
        "sample_n": 1500,
        "label": "FLASH",
        "format_fn": "arc_mcq",
        "rationale": "ArcMMLU flash-lite=0.82; ARC-Easy same domain different questions",
    },
    {
        "ra_datasets": ["ArcMMLU"],
        "hf_path": "allenai/ai2_arc",
        "hf_name": "ARC-Challenge",
        "split": "train",
        "sample_n": 1000,
        "label": "FLASH",
        "format_fn": "arc_mcq",
        "rationale": "ArcMMLU flash-lite=0.82; ARC-Challenge same domain",
    },
    # MMLUPro_* → MMLU-Pro (test split has more examples than validation)
    {
        "ra_datasets": [
            "MMLUPro_computer science",
            "MMLUPro_engineering",
            "MMLUPro_biology",
            "MMLUPro_chemistry",
            "MMLUPro_physics",
            "MMLUPro_math",
            "MMLUPro_history",
            "MMLUPro_economics",
            "MMLUPro_business",
            "MMLUPro_law",
            "MMLUPro_psychology",
            "MMLUPro_health",
            "MMLUPro_philosophy",
        ],
        "hf_path": "TIGER-Lab/MMLU-Pro",
        "hf_name": None,
        "split": "test",
        "sample_n": 3000,
        "label": "FLASH",
        "format_fn": "mmlupro_mcq",
        "rationale": "All MMLUPro subjects flash-lite ≥0.74; test split has 12k examples",
    },
    # MMLU → MMLU (train split, multiple subjects)
    {
        "ra_datasets": ["MMLU_formal_logic", "MMLU_management"],
        "hf_path": "cais/mmlu",
        "hf_name": "all",
        "split": "validation",
        "sample_n": 2000,
        "label": "FLASH",
        "format_fn": "mmlu_mcq",
        "rationale": "MMLU_formal_logic flash-lite=0.82, management=0.89",
    },
    # PubMedQA → MedQA USMLE (different dataset, same biomedical domain)
    {
        "ra_datasets": ["PubMedQA"],
        "hf_path": "GBaker/MedQA-USMLE-4-options",
        "hf_name": None,
        "split": "train",
        "sample_n": 2000,
        "label": "FLASH",
        "format_fn": "medqa_mcq",
        "rationale": "PubMedQA flash-lite=0.79; MedQA-USMLE same biomedical MCQ domain",
    },
    # MedMCQA → HeadQA (Spanish medical MCQ, different dataset same domain)
    {
        "ra_datasets": ["MedMCQA"],
        "hf_path": "openlifescienceai/medmcqa",
        "hf_name": None,
        "split": "train",
        "sample_n": 2000,
        "label": "FLASH",
        "format_fn": "medmcqa_mcq",
        "rationale": "MedMCQA flash-lite=0.83; same dataset different split",
    },
    # GeoBench → MMLU high_school_geography + world_religions (same knowledge domain)
    {
        "ra_datasets": ["GeoBench", "GeoGraphyData_100k"],
        "hf_path": "cais/mmlu",
        "hf_name": "high_school_geography",
        "split": "test",
        "sample_n": 200,
        "label": "FLASH",
        "format_fn": "mmlu_mcq",
        "rationale": "GeoBench flash-lite=0.91; MMLU geography is similar knowledge MCQ",
    },
    # OpenTDB → TriviaQA (similar trivia format)
    {
        "ra_datasets": [
            "OpenTDB_General Knowledge",
            "OpenTDB_Science: Computers",
            "OpenTDB_Geography",
            "OpenTDB_History",
            "OpenTDB_Animals",
            "OpenTDB_Art",
            "OpenTDB_Celebrities",
            "OpenTDB_Sports",
            "OpenTDB_Vehicles",
            "OpenTDB_Entertainment: Books",
            "OpenTDB_Entertainment: Film",
            "OpenTDB_Entertainment: Music",
            "OpenTDB_Entertainment: Television",
            "OpenTDB_Entertainment: Video Games",
            "OpenTDB_Entertainment: Board Games",
            "OpenTDB_Entertainment: Cartoon & Animations",
            "OpenTDB_Entertainment: Comics",
            "OpenTDB_Entertainment: Japanese Anime & Manga",
            "OpenTDB_Entertainment: Musicals & Theatres",
            "OpenTDB_Science & Nature",
            "OpenTDB_Science: Mathematics",
        ],
        "hf_path": "trivia_qa",
        "hf_name": "unfiltered",
        "split": "train",
        "sample_n": 2000,
        "label": "FLASH",
        "format_fn": "triviaqa",
        "rationale": "All OpenTDB flash-lite ≥0.86; trivia is cheap-model territory",
    },
    # QANTA → quiz bowl / adversarial factual Q&A (NODATA — QWEN235B)
    # jeopardy + trivia_qa/rc both have download issues; use TruthfulQA instead:
    #   TruthfulQA is designed to expose model failures on hard factual questions,
    #   making it a good signal for QWEN235B (only best model handles tricky facts).
    {
        "ra_datasets": [
            "QANTA_Literature",
            "QANTA_History",
            "QANTA_Science",
            "QANTA_Fine Arts",
            "QANTA_Philosophy",
            "QANTA_Social Science",
            "QANTA_Geography",
        ],
        "hf_path": "truthful_qa",
        "hf_name": "multiple_choice",
        "split": "validation",
        "sample_n": 700,
        "label": "QWEN235B",
        "format_fn": "truthfulqa_mcq",
        "rationale": "QANTA all models <0.45; TruthfulQA is adversarial factual → best model",
    },
    # Ethics → MMLU moral/philosophy MCQs (hendrycks/ethics uses deprecated loading script)
    {
        "ra_datasets": ["Ethics_commonsense", "Ethics_virtue", "Ethics_deontology"],
        "hf_path": "cais/mmlu",
        "hf_name": "moral_disputes",
        "split": "validation",
        "sample_n": 500,
        "label": "FLASH",
        "format_fn": "mmlu_mcq",
        "rationale": "Ethics_commonsense/virtue/deontology flash≥0.90; moral MCQ is flash territory",
    },
    # Ethics_justice → DEEPSEEK: use professional_law (harder ethical/legal MCQs)
    {
        "ra_datasets": ["Ethics_justice"],
        "hf_path": "cais/mmlu",
        "hf_name": "professional_law",
        "split": "validation",
        "sample_n": 500,
        "label": "DEEPSEEK",
        "format_fn": "mmlu_mcq",
        "rationale": "Ethics_justice flash=0.24; professional_law requires careful reasoning → deepseek",
    },
    # SocialiQA → CommonsenseQA (same commonsense reasoning domain)
    {
        "ra_datasets": ["SocialiQA"],
        "hf_path": "tau/commonsense_qa",
        "hf_name": None,
        "split": "train",
        "sample_n": 1000,
        "label": "FLASH",
        "format_fn": "commonsenseqa_mcq",
        "rationale": "SocialiQA flash-lite=0.72; commonsense MCQ is flash territory",
    },
    # NarrativeQA → QuALITY (long-form reading comprehension)
    {
        "ra_datasets": ["NarrativeQA"],
        "hf_path": "emozilla/quality",
        "hf_name": None,
        "split": "train",
        "sample_n": 1000,
        "label": "FLASH",
        "format_fn": "narrative_qa",
        "rationale": "NarrativeQA flash-lite=0.56 (marginal but still cheapest correct)",
    },
    # MusicTheoryBench → world_religions MCQ (music_theory not in cais/mmlu configs)
    {
        "ra_datasets": ["MusicTheoryBench"],
        "hf_path": "cais/mmlu",
        "hf_name": "world_religions",
        "split": "test",
        "sample_n": 200,
        "label": "FLASH",
        "format_fn": "mmlu_mcq",
        "rationale": "MusicTheoryBench flash=0.64; world_religions is same niche-knowledge MCQ domain",
    },
    # SuperGLUE tasks
    {
        "ra_datasets": [
            "SuperGLUE-Wic",
            "SuperGLUE-Entailment",
            "SuperGLUE-CausalReasoning",
            "SuperGLUE-Wsc",
        ],
        "hf_path": "super_glue",
        "hf_name": "wic",
        "split": "train",
        "sample_n": 500,
        "label": "FLASH",
        "format_fn": "superglue_wic",
        "rationale": "SuperGLUE-Wic flash-lite=0.78; word-in-context is flash territory",
    },
    {
        "ra_datasets": ["SuperGLUE-RC"],
        "hf_path": "super_glue",
        "hf_name": "record",
        "split": "train",
        "sample_n": 500,
        "label": "QWEN235B",
        "format_fn": "superglue_record",
        "rationale": "SuperGLUE-RC flash=0.30, only qwen235b=0.65 passes threshold",
    },
    {
        "ra_datasets": ["SuperGLUE-QA"],
        "hf_path": "super_glue",
        "hf_name": "multirc",
        "split": "train",
        "sample_n": 500,
        "label": "DEEPSEEK",
        "format_fn": "superglue_multirc",
        "rationale": "SuperGLUE-QA flash=0.46, deepseek=0.70 — deepseek wins",
    },
    {
        "ra_datasets": ["SuperGLUE-ClozeTest"],
        "hf_path": "super_glue",
        "hf_name": "copa",
        "split": "train",
        "sample_n": 400,
        "label": "QWEN235B",
        "format_fn": "superglue_copa",
        "rationale": "SuperGLUE-ClozeTest all models <0.50; use best model",
    },
    # Math: GSM8K, AsDiv, MathQA → FLASH (flash-lite handles these)
    {
        "ra_datasets": ["GSM8K", "AsDiv"],
        "hf_path": "openai/gsm8k",
        "hf_name": "main",
        "split": "train",
        "sample_n": 1000,
        "label": "FLASH",
        "format_fn": "gsm8k_math",
        "rationale": "GSM8K flash-lite=0.70; elementary math is flash territory",
    },
    # MathQA (math_qa deprecated) → skip; GSM8K (above) already covers flash-level math.
    # hendrycks_math entries removed: their surface form is identical to AIME (DEEPSEEK),
    # causing the MLP to confuse competition math difficulty tiers (84% DEEPSEEK recall).
    # MATH (hendrycks/competition_math doesn't exist on Hub) → skip for same reason.
    # AIME → DEEPSEEK (flash-lite=0.35, deepseek=0.72)
    # Use AI-MO/aimo-validation-aime which is on the hub
    {
        "ra_datasets": ["AIME"],
        "hf_path": "AI-MO/aimo-validation-aime",
        "hf_name": None,
        "split": "train",
        "sample_n": 300,
        "label": "DEEPSEEK",
        "format_fn": "aime_math",
        "rationale": "AIME flash-lite=0.35, deepseek=0.72 — olympiad math needs deepseek",
    },
    # LiveCodeBench → HumanEval / MBPP (coding)
    {
        "ra_datasets": ["LiveCodeBench"],
        "hf_path": "openai/openai_humaneval",
        "hf_name": None,
        "split": "test",
        "sample_n": 164,
        "label": "FLASH",
        "format_fn": "humaneval_code",
        "rationale": "LiveCodeBench flash-lite=0.67 (surprisingly flash handles most coding MCQs)",
    },
    # mbpp/sanitized has no train split → use full config which has train
    {
        "ra_datasets": ["LiveCodeBench"],
        "hf_path": "google-research-datasets/mbpp",
        "hf_name": "full",
        "split": "train",
        "sample_n": 374,
        "label": "FLASH",
        "format_fn": "mbpp_code",
        "rationale": "LiveCodeBench flash=0.67; mbpp/full train split has same examples as sanitized",
    },
    # ChessInstruct → NODATA → QWEN235B
    {
        "ra_datasets": ["ChessInstruct"],
        "hf_path": "Lichess/chess-puzzles",
        "hf_name": None,
        "split": "train",
        "sample_n": 500,
        "label": "QWEN235B",
        "format_fn": "chess_generic",
        "rationale": "ChessInstruct flash=0.34 deepseek=0.41 — both fail; use best model",
    },
    # FinQA → NODATA → QWEN235B (flare-finqa is similar financial text reasoning)
    {
        "ra_datasets": ["FinQA"],
        "hf_path": "ChanceFocus/flare-finqa",
        "hf_name": None,
        "split": "train",
        "sample_n": 500,
        "label": "QWEN235B",
        "format_fn": "finqa_generic",
        "rationale": "FinQA flash=0.36 deepseek=0.33 — both fail; financial reasoning needs best",
    },
    # WMT19 translation → FLASH for common pairs, DEEPSEEK for cs-en, QWEN235B for rare
    {
        "ra_datasets": [
            "WMT19-de-en",
            "WMT19-zh-en",
            "WMT19-fi-en",
            "WMT19-gu-en",
            "WMT19-ru-en",
        ],
        "hf_path": "wmt/wmt14",
        "hf_name": "de-en",
        "split": "train",
        "sample_n": 1000,
        "label": "FLASH",
        "format_fn": "wmt_translation",
        "rationale": "WMT19-de-en flash=0.72; common language pairs flash handles fine",
    },
    {
        "ra_datasets": ["WMT19-cs-en"],
        "hf_path": "wmt/wmt16",
        "hf_name": "cs-en",
        "split": "train",
        "sample_n": 500,
        "label": "DEEPSEEK",
        "format_fn": "wmt_translation",
        "rationale": "WMT19-cs-en flash=0.48 (below threshold), deepseek=0.50 — deepseek territory",
    },
    {
        "ra_datasets": ["WMT19-kk-en", "WMT19-lt-en"],
        "hf_path": "wmt/wmt19",
        "hf_name": "lt-en",
        "split": "train",
        "sample_n": 300,
        "label": "QWEN235B",
        "format_fn": "wmt_translation",
        "rationale": "WMT19-lt-en/kk-en flash=0.35/0.31 — rare lang pairs need best model",
    },
    # OpenBookQA → FLASH (science knowledge)
    {
        "ra_datasets": ["OpenTDB_Science & Nature", "OpenTDB_Science: Computers"],
        "hf_path": "allenai/openbookqa",
        "hf_name": "main",
        "split": "train",
        "sample_n": 500,
        "label": "FLASH",
        "format_fn": "arc_mcq",
        "rationale": "Science knowledge is flash territory; openbookqa is similar domain",
    },
    # WinoGrande → FLASH (commonsense)
    {
        "ra_datasets": ["SuperGLUE-Wsc"],
        "hf_path": "winogrande",
        "hf_name": "winogrande_xl",
        "split": "train",
        "sample_n": 1000,
        "label": "FLASH",
        "format_fn": "winogrande",
        "rationale": "SuperGLUE-Wsc flash=1.00; commonsense coreference is flash territory",
    },
]

if __name__ == "__main__":
    from collections import Counter

    label_counts = Counter(d["label"] for d in DOMAIN_MAP)
    total_samples = sum(d["sample_n"] for d in DOMAIN_MAP)
    print(f"Total dataset configs: {len(DOMAIN_MAP)}")
    print(f"Total max samples: {total_samples:,}")
    print(f"Label distribution: {dict(label_counts)}")
    print()
    for label in ["FLASH", "DEEPSEEK", "QWEN235B"]:
        configs = [d for d in DOMAIN_MAP if d["label"] == label]
        print(
            f"{label}: {len(configs)} configs, {sum(d['sample_n'] for d in configs):,} samples"
        )
        for c in configs:
            print(
                f"  {c['hf_path']}/{c.get('hf_name') or ''} ({c['split']}, n={c['sample_n']})"
            )
        print()
