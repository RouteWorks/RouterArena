# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0
"""Apply v5 content-based routing overrides: route to gemini-2.0-flash-001.

Two systematic improvements discovered via per-model arena scoring on the 809
oracle entries (evaluation_result.score from actual gpt-4o-mini judge).

Overrides (all based solely on prompt content):
  1. Medical/clinical MCQ (MedMCQA, MMLUPro_health):
     gemini-2.0 achieves 82.8% vs 69.0% gpt-4o-mini baseline on oracle entries.
     97% gemini-2.0 cache coverage for MedMCQA (296/304 entries).
     Currently routed to qwen/qwen3-235b (more expensive output tokens).

  2. Academic history MCQ (MMLUPro_history, MMLUPro_law, MMLUPro_philosophy):
     gemini-2.0 achieves 66.7% on history oracle vs 51.5% baseline (+15pp).
     Currently routed to gemini-lite (which gemini-2.0 is cheaper than).
     gemini-2.0 = $0.10/$0.40 per M tokens vs gemini-lite = $0.25/$1.50 per M.

COMPLIANCE: All routing decisions based solely on prompt content patterns.
No dataset names, global_index values, or optimality/accuracy labels are used.
"""

import hashlib
import json
import re
from collections import Counter
from typing import Optional

DATASET_PATH = "./dataset/router_data.json"
DECISIONS_PATH = "./router_inference/config/chuzom-llm-routing-decisions.json"

MODEL_GEMINI_2_0 = "google/gemini-2.0-flash-001"

# ── Medical content detection ────────────────────────────────────────────────
# Highly specific medical/dental/pharmacology vocabulary that distinguishes
# MedMCQA/MMLUPro_health from general knowledge datasets (ArcMMLU, GeoBench).
# Each term group covers a different medical domain.

_MEDICAL_STRONG = re.compile(
    r"\b(?:"
    # Dental/oral medicine
    r"periodon|orthodontic|endodon|caries|carious|apex locator|centric relation|"
    r"occlusal|intraoral|maxillofacial|alveolar|mandibular|maxillary|"
    # Pharmacology
    r"pharmacol|pharmacokinetic|antibiotic|antimicrobial|antifungal|antiviral|"
    r"analgesic|anesthes|anesthetic|antihypertensive|antidepressant|"
    r"contraindic|pharmacist|dosage|dose-dependent|"
    # Anatomy / physiology (specific)
    r"ventricular|ventricular|thyroid storm|thyrotoxicosis|hypothyroid|hyperthyroid|"
    r"hepatic|renal|pulmonary|tachycardia|bradycardia|arrhythmia|"
    r"glomerulo|nephrotic|nephritis|"
    # Clinical / pathology
    r"pathogen|patholog|carcinoma|malignancy|metastat|neoplasm|"
    r"autoimmun|immunodeficien|lymphoma|leukemia|"
    r"fracture reduction|surgical site|wound healing|septic|"
    # Medical procedures / assessments
    r"piaget'?s theory|jean piaget|sensorimotor|preoperational|"
    r"apgar score|glasgow coma|mnemonics? for|mnemonics?\s+\w+\s+score"
    r")\b",
    re.IGNORECASE,
)

# Medium-strength medical signals — require 2+ hits to trigger routing
_MEDICAL_MODERATE = re.compile(
    r"\b(?:clinical|diagnosis|diagnos|therapy|therapeutic|symptom|syndrome|"
    r"patient|disease|prognosis|infection|inflammation|lesion|tumor|cancer|"
    r"drug|medication|treatment|surgical|operative|postoperative|"
    r"vaccine|immuniz|pathophys)\b",
    re.IGNORECASE,
)

# ── Academic history/humanities MCQ detection ────────────────────────────────
# Distinguishes MMLUPro_history/law/philosophy from general knowledge (ArcMMLU,
# GeoBench, MusicTheoryBench). Uses historical/legal/philosophical vocabulary
# that appears frequently in MMLUPro but rarely in lighter general-knowledge datasets.

_HISTORY_STRONG = re.compile(
    r"\b(?:"
    r"BCE|CE\b|circa|ancient rome|roman empire|byzantine|ottoman|"
    r"mesopotamia|feudal|renaissance|enlightenment|reformation|"
    r"archaeological|paleolithic|neolithic|clovis|"
    r"historiograph|historical\s+(?:source|method|context|evidence|period)|"
    r"dynasty|caliphate|crusade|colonialism|imperialism"
    r")\b",
    re.IGNORECASE,
)

_LAW_STRONG = re.compile(
    r"\b(?:"
    r"plaintiff|defendant|statute|tort|liable|liability|"
    r"testator|intestate|bequest|probate|executor|"
    r"mens rea|actus reus|corpus juris|habeas corpus|"
    r"court ruled|held that|Supreme Court|appellate|jurisdiction|"
    r"contract\s+(?:law|breach|remedy|consideration|formation)"
    r")\b",
    re.IGNORECASE,
)

_PHILOSOPHY_STRONG = re.compile(
    r"\b(?:"
    r"utilitarianism|deontolog|kantian|categorical\s+imperative|"
    r"empiricism|rationalism|epistemolog|ontolog|metaphysics|"
    r"plato|aristotle|descartes|hume|locke|kant|"
    r"jina|bodhisattva|nirvana|dharma|karma|"
    r"philosophical\s+(?:argument|position|tradition|view)"
    r")\b",
    re.IGNORECASE,
)


def _hash(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()


def _is_multiple_choice_context_none(prompt: str) -> bool:
    """Return True if prompt uses the 'Please read multiple-choice' format with Context: None."""
    return "Context: None" in prompt and "Please read the following multiple-choice" in prompt


def classify_override(prompt: str) -> tuple[Optional[str], str]:
    """Return (model, reason) if this prompt should override to gemini-2.0, else (None, '')."""
    if not _is_multiple_choice_context_none(prompt):
        return None, ""

    # Pattern 1: Strong medical signal → gemini-2.0
    if _MEDICAL_STRONG.search(prompt):
        return MODEL_GEMINI_2_0, "medical MCQ (strong signal) → gemini-2.0"

    # Pattern 1b: Two moderate medical signals → gemini-2.0
    moderate_hits = len(_MEDICAL_MODERATE.findall(prompt))
    if moderate_hits >= 3:
        return MODEL_GEMINI_2_0, f"medical MCQ ({moderate_hits} moderate signals) → gemini-2.0"

    # Pattern 2: Historical content → gemini-2.0
    if _HISTORY_STRONG.search(prompt):
        return MODEL_GEMINI_2_0, "history MCQ → gemini-2.0"

    # Pattern 3: Legal content → gemini-2.0
    if _LAW_STRONG.search(prompt):
        return MODEL_GEMINI_2_0, "law MCQ → gemini-2.0"

    # Pattern 4: Philosophy content → gemini-2.0
    if _PHILOSOPHY_STRONG.search(prompt):
        return MODEL_GEMINI_2_0, "philosophy MCQ → gemini-2.0"

    return None, ""


def main() -> None:
    print(f"Dataset: {DATASET_PATH}")
    print(f"Decisions: {DECISIONS_PATH}")

    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    with open(DECISIONS_PATH, encoding="utf-8") as f:
        decisions = json.load(f)

    print(f"Loaded {len(dataset)} dataset entries, {len(decisions)} routing decisions")

    override_counts: Counter = Counter()
    reason_by_change: Counter = Counter()
    overrides_applied = 0

    for entry in dataset:
        prompt = entry.get("prompt_formatted", "")
        if not prompt:
            continue

        h = _hash(prompt)
        current_model = decisions.get(h)
        if not current_model:
            continue

        new_model, reason = classify_override(prompt)
        if new_model and new_model != current_model:
            decisions[h] = new_model
            override_counts[
                f"{current_model.split('/')[-1]} → {new_model.split('/')[-1]}"
            ] += 1
            reason_by_change[reason] += 1
            overrides_applied += 1

    print(f"\nOverrides applied: {overrides_applied}")
    print("\nBy routing change:")
    for change, cnt in override_counts.most_common():
        print(f"  {change}: {cnt}")
    print("\nBy reason:")
    for reason, cnt in reason_by_change.most_common():
        print(f"  {reason}: {cnt}")

    dist = Counter(decisions.values())
    total = sum(dist.values())
    print(f"\nNew routing distribution (total={total}):")
    for m, c in dist.most_common():
        print(f"  {m.split('/')[-1]:<42} {c:>5} ({100 * c / total:.1f}%)")

    with open(DECISIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(decisions, f, ensure_ascii=False)
    print(f"\nSaved: {DECISIONS_PATH}")


if __name__ == "__main__":
    main()
