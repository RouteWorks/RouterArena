# RouterArena official evaluator — off-laptop (k8s / Linux container)

The official `llm_evaluation/run.py` uses Python multiprocessing + a code-execution
sandbox that CRASHES on macOS (`resource_tracker` semaphore leak). It runs fine on
Linux. This packages it to run in a container / k8s Job.

## Build & run (local Linux container)
    docker buildx build --platform linux/amd64 -t <img> .
    docker run -e ROUTER=<router-name> -e SPLIT=sub_10 \
      -v $PWD/../../router_inference/predictions/<router>.json:/app/router_inference/predictions/<router>.json <img>

## k8s Job (cruq namespace, dockerhub-secret pull secret)
    kubectl apply -f job.yaml   # set ROUTER/SPLIT env; reads image from Docker Hub

## IMPORTANT caveat (measured 2026-08-23)
This local reproduction does NOT faithfully match RouterArena's official scoring for a
large class of datasets. Validation: scoring the #1 leaderboard router (Paix2) through
this harness yields 0.475 on sub_10 vs its official 79.69% on full — because the sub_10
prediction files' global-index keys do not all match the evaluator's full-arrow
`all_data`, so `_get_ground_truth` silently returns None and those datasets (AsDiv,
FinQA, QANTA, WMT19, SuperGLUE, ...) score 0 for EVERY router. Treat local official
numbers as unreliable for those families; the only trustworthy leaderboard number comes
from RouterArena's own `/evaluate` PR workflow. The lightweight boxed-match grader
(`router_evaluation/lightweight_grade.py`) remains a fair proxy for the MCQ-heavy subset.

Files: Dockerfile (deps + eval), summarize.py (aggregates per-dataset + arena-S),
job.yaml (k8s Job). Bakes only llm_evaluation/, global_utils/, universal_model_names.py,
model_cost/, dataset/routerarena{,_10}; predictions are mounted or baked per run.
