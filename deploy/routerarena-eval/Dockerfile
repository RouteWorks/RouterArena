FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=/app
WORKDIR /app
RUN pip install --no-cache-dir \
    "datasets>=4.0.0" "pandas>=2.2" "numpy>=2.0" "pyarrow>=17" \
    rouge sympy latex2sympy2 fuzzywuzzy python-Levenshtein jieba regex \
    python-dotenv "huggingface-hub>=0.34"
COPY . /app
# default: score the router named by ROUTER env on SPLIT, then print a summary
CMD ["/bin/sh","-c","python llm_evaluation/run.py \"$ROUTER\" \"$SPLIT\" ; python summarize.py \"$ROUTER\""]
