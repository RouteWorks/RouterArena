#!/usr/bin/env bash
# ci_quarantine_guard.sh — RouterArena submission integrity check
#
# Asserts that no quarantined RA-tainted artifacts are read during prediction
# generation. Run this before any `generate_prediction_file.py` invocation.
#
# Quarantined artifacts (see ~/.routerarena-quarantine/):
#   - chuzom-llm-routing-decisions.json  (8400-entry RA-oracle lookup table)
#   - chuzom-domain-classifier.joblib    (Gate 0 classifier, RA-accuracy labels)
#   - chuzom-proxy-classifier.joblib     (Gate 0 proxy, RA-accuracy labels)
#   - llm_evaluation_dict.json           (per-query RA score oracle)
#   - .self_consistency_cache*.json      (RA-keyed cache files)
#   - domain_dataset_map.py              (RA-accuracy-derived dataset labels)
#   - build_outcome_labels.py            (RA-accuracy-derived label builder)
#   - build_domain_classifier.py         (classifier trained on RA labels)
#
# Usage: bash scripts/ci_quarantine_guard.sh [--strict]
#   --strict: exit 1 if any quarantined file references found in source

set -euo pipefail

STRICT=0
for arg in "$@"; do
    [[ "$arg" == "--strict" ]] && STRICT=1
done

QUARANTINED_PATTERNS=(
    "chuzom-llm-routing-decisions\.json"
    "chuzom-domain-classifier\.joblib"
    "chuzom-proxy-classifier\.joblib"
    "llm_evaluation_dict\.json"
    "self_consistency_cache"
    "domain_dataset_map\.py"
    "build_outcome_labels\.py"
    "build_domain_classifier\.py"
    "apply_v[0-9]\+_"
    "PLAN_TOP3\.md"
)

SEARCH_DIRS=(
    "router_inference/router/chuzom_v3_router.py"
    "router_inference/config/chuzom-v3.json"
    "scripts/generate_chuzom_v3_predictions.py"
)

VIOLATIONS=0

echo "=== Chuzom v3 Quarantine Guard ==="
echo "Checking source files for references to quarantined RA artifacts..."
echo ""

SELF=$(basename "$0")
for pattern in "${QUARANTINED_PATTERNS[@]}"; do
    matches=$(grep -r --include="*.py" --include="*.json" --include="*.sh" \
        -l "$pattern" "${SEARCH_DIRS[@]}" 2>/dev/null | grep -v "/$SELF$" || true)
    if [[ -n "$matches" ]]; then
        echo "VIOLATION: '$pattern' referenced in:"
        echo "$matches" | sed 's/^/  /'
        VIOLATIONS=$((VIOLATIONS + 1))
    fi
done

echo ""
echo "=== File presence check ==="
TAINTED_FILES=(
    "router_inference/config/chuzom-domain-classifier.joblib"
    "router_inference/config/chuzom-proxy-classifier.joblib"
    "router_inference/config/chuzom-llm-routing-decisions.json"
    "router_inference/llm_evaluation_dict.json"
    ".self_consistency_cache.json"
    ".self_consistency_cache_robust.json"
    "scripts/domain_dataset_map.py"
    "scripts/build_outcome_labels.py"
    "scripts/build_domain_classifier.py"
    "PLAN_TOP3.md"
)

for f in "${TAINTED_FILES[@]}"; do
    if [[ -f "$f" ]]; then
        echo "VIOLATION: Tainted file present in working tree: $f"
        VIOLATIONS=$((VIOLATIONS + 1))
    fi
done

echo ""
if [[ $VIOLATIONS -eq 0 ]]; then
    echo "✓ All $((${#QUARANTINED_PATTERNS[@]} + ${#TAINTED_FILES[@]})) quarantine checks passed."
    exit 0
else
    echo "✗ $VIOLATIONS quarantine violation(s) found."
    if [[ $STRICT -eq 1 ]]; then
        exit 1
    else
        echo "  (Run with --strict to fail the build on violations)"
        exit 0
    fi
fi
