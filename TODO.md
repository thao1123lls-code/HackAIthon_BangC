# TODO - BTC RAG Docker + Predict Refactor

## Step 1 — Analyze & implement required changes
- [x] Understand repo structure (Dockerfile/main.py/rag.py)

## Step 2 — Dockerfile requirements (mandatory)
- [x] Switch base image to `nvidia/cuda:12.2.0-devel-ubuntu20.04`
- [x] Set `WORKDIR /code`
- [x] `COPY requirements.txt` then `pip install -r requirements.txt`
- [x] `COPY . /code`
- [x] Container entrypoint: `CMD ["bash", "/code/inference.sh"]`

## Step 3 — Create `predict.py`
- [x] Port logic from `main.py`
- [x] Read input fixed path `/code/private_test.json`
- [x] Robust nested JSON parsing to find `qid` + `question`
- [x] Per-sample inference time measurement (loop per sample)
- [x] Export:
  - [x] `/code/output/submission.csv` (qid,answer)
  - [x] `/code/output/submission_time.csv` (qid,answer,time)

## Step 4 — Create `inference.sh`
- [x] Orchestrator script calling `python3 /code/predict.py`

## Step 5 — Update README
- [x] Rewrite `README.md` with required 3 sections:
  - [x] Pipeline Flow (RAG 2 chặng)
  - [x] Data Processing
  - [x] Resource Initialization (Vector DB indexing + GPU model load)


## Step 6 — Validate
- [ ] Smoke test by running `python predict.py` (if possible)
- [ ] Docker build + run to confirm 2 CSV outputs exist with correct columns

