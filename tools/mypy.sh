#!/bin/bash

# Some code is borrowed from kvcached. Thanks!

CI=${1:-0}
PYTHON_VERSION=${2:-local}

if [[ "$CI" -eq 1 ]]; then
    set -e
fi

if [[ "$PYTHON_VERSION" == "local" ]]; then
    PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
fi

EXCLUDE_PATTERN=''

run_mypy() {
    local target=$1; shift || true
    # Default to the current directory (full repo) if no target is specified.
    if [[ -z "$target" ]]; then
        target="."
    fi

    echo "Running mypy on $target"

    # Build mypy arguments, adding --exclude only when non-empty.
    local mypy_args=(--python-version "${PYTHON_VERSION}" --namespace-packages)
    if [[ -n "${EXCLUDE_PATTERN}" ]]; then
        mypy_args+=(--exclude "${EXCLUDE_PATTERN}")
    fi

    if [[ "$CI" -eq 1 ]]; then
        # In CI, run mypy with full strictness.
        mypy "${mypy_args[@]}" "$@" "$target"
    else
        # Local runs are a bit more lenient and skip heavy import following.
        mypy_args+=(--follow-imports skip)
        mypy "${mypy_args[@]}" "$@" "$target"
    fi
}

run_mypy llm_evaluation
run_mypy llm_inference
run_mypy router_inference
run_mypy scripts
run_mypy universal_model_names.py
