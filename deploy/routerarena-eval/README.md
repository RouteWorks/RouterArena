# RouterArena official evaluator — off-laptop (k8s / Linux container)

The official `llm_evaluation/run.py` uses Python multiprocessing + a code-execution
sandbox that CRASHES on macOS (`resource_tracker` semaphore leak). It runs fine on
Linux. This packages it to run in a container / k8s Job. Image: `talentreviewai/routerarena-eval:v3`.

## Build inputs (the build context MUST contain, next to the Dockerfile)
    llm_evaluation/  global_utils/  universal_model_names.py  model_cost/
    config/eval_config/            # REQUIRED — per-dataset scorer configs (see fix below)
    dataset/routerarena/           # full 8400 arrow (ground truth for full)
    dataset/routerarena_10/        # sub_10 arrow (ground truth for sub_10)
    router_inference/predictions/  # the prediction file(s) to score (or mount at run)

## Build & run (local Linux container)
    docker buildx build --platform linux/amd64 -t talentreviewai/routerarena-eval:v3 .
    docker run -e ROUTER=<router-name> -e SPLIT=sub_10 \
      -v $PWD/pred/<router>.json:/app/router_inference/predictions/<router>.json \
      talentreviewai/routerarena-eval:v3

## k8s Job (cruq namespace, dockerhub-secret pull secret)
    kubectl apply -f job.yaml      # set ROUTER/SPLIT env; pulls image from Docker Hub

## The fix that made local scoring trustworthy (2026-08-23)
Earlier images OMITTED `config/eval_config/`. Without it, `load_eval_config_for_dataset`
returns no metrics and the evaluator silently falls back to `mcq_accuracy` for EVERY
dataset — so numeric/translation/word-sense answers (AsDiv, FinQA, MATH-family, WMT19,
SuperGLUE, AIME) were scored as multiple-choice and got 0 for ALL routers. Validation:
scoring the #1 router `Paix2-router.json` on sub_10 gave 0.475 (broken) → 0.52+ (fixed),
with AsDiv 0→0.43, FinQA 0→0.14, WMT19 0→0.41, SuperGLUE-Wic 0→0.50 recovering. Always
include `config/eval_config/` in the build context. (QANTA zeros are genuine — models
ramble instead of naming the gold entity; that is real, not a harness bug.)

Note: `dataset/livecodebench` (2.9 GB) is intentionally excluded; the 38 LiveCodeBench
sub_10 items are skipped. Add it only if you need code scoring.
