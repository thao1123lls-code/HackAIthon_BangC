"""HackAIthon 2026 Bảng C - AI Multiple Choice QA System

Entry point for grading.
Requirements (BTC):
- Read input from fixed path: /code/private_test.json
- Measure per-sample inference time (each question)
- Output 2 CSV files:
  - /code/output/submission.csv: qid, answer
  - /code/output/submission_time.csv: qid, answer, time
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kwargs: x

try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    print(f"Failed to import vLLM: {e}")
    print("This system requires GPU support. Install with: pip install vllm")
    sys.exit(1)

from rag import RAGSystem


# ==================== CONSTANTS ====================
MODEL_NAME = "Qwen/Qwen1.5-7B-Chat"
GPU_MEMORY_UTIL = 0.9
MAX_MODEL_LEN = 2048
MAX_TOKENS = 256
TEMPERATURE = 0.0
DEFAULT_ANSWER = "A"
KB_PATH = "knowledge_base.txt"
RAG_TOP_K = 2

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Pre-compile regex patterns for performance
REGEX_PATTERNS = {
    "primary": re.compile(r"Đáp án:\\s*([ABCD])", re.IGNORECASE),
    "keywords": re.compile(r"(chọn|là)\\s*([ABCD])", re.IGNORECASE),
    "last_choice": re.compile(r"\\b([ABCD])\\b"),
}


def extract_answer(generated_text: str) -> str:
    """Extract answer from generated text with 3-layer fallback strategy."""
    if not generated_text or not isinstance(generated_text, str):
        return DEFAULT_ANSWER

    match = REGEX_PATTERNS["primary"].search(generated_text)
    if match:
        return match.group(1).upper()

    match_2 = REGEX_PATTERNS["keywords"].search(generated_text)
    if match_2:
        return match_2.group(2).upper()

    matches = REGEX_PATTERNS["last_choice"].findall(generated_text)
    if matches:
        return matches[-1].upper()

    logger.warning(
        "Could not extract answer, using default '%s'. Text head: %s",
        DEFAULT_ANSWER,
        generated_text[:100],
    )
    return DEFAULT_ANSWER


def _iter_candidates(obj: Any) -> Iterable[Dict[str, Any]]:
    """Yield dict objects that may contain qid/question keys (including nested)."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_candidates(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_candidates(item)


def load_private_json(path: str) -> List[Dict[str, Any]]:
    """Robustly load /code/private_test.json.

    Expected: list of items OR dict with nested list.
    Fallback: scan nested objects for keys containing qid + question.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Case 1: raw is list
    if isinstance(raw, list):
        items: List[Dict[str, Any]] = []
        for x in raw:
            if isinstance(x, dict):
                items.append(x)
        return items

    # Case 2: raw is dict; try common containers
    if isinstance(raw, dict):
        for k in ["data", "items", "questions", "samples", "test", "test_data"]:
            v = raw.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]

    # Case 3: deep scan dicts for qid + question
    found: List[Dict[str, Any]] = []
    for cand in _iter_candidates(raw):
        if "qid" in cand and ("question" in cand or "query" in cand):
            found.append(cand)

    return found


def normalize_samples(samples: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Return list of (qid, question) after cleaning."""
    normalized: List[Tuple[str, str]] = []
    for s in samples:
        if not isinstance(s, dict):
            continue

        qid = s.get("qid")
        question = s.get("question", s.get("query"))

        if qid is None or question is None:
            continue

        qid_str = str(qid)
        q_str = str(question).strip()
        if not q_str:
            continue

        normalized.append((qid_str, q_str))

    if not normalized:
        raise ValueError("No valid samples found with keys qid and question")

    return normalized


def prepare_prompt(question: str, context: str) -> str:
    context_block = f"[THÔNG TIN THAM KHẢO]:\n{context}\n\n" if context else ""
    prompt = f"""Bạn là một hệ thống AI xuất sắc trong việc thi trắc nghiệm.
{context_block}Nhiệm vụ của bạn là giải quyết câu hỏi dưới đây một cách logic và cẩn thận:
{question}

--- QUY TRÌNH SUY LUẬN BẮT BUỘC ---
1. Phân tích ngữ cảnh và yêu cầu của câu hỏi.
2. Đánh giá ngắn gọn từng phương án A, B, C, D. Loại trừ các phương án sai.
3. Chốt lại đáp án đúng nhất.
Dòng cuối cùng BẮT BUỘC có định dạng: "Đáp án: X" (trong đó X là A, B, C hoặc D).
Không in thêm bất kỳ ký tự nào sau dòng đáp án."""
    return prompt


def main() -> bool:
    try:
        input_path = "/code/private_test.json"
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file missing: {input_path}")
        output_dir = "/code/output"
        os.makedirs(output_dir, exist_ok=True)

        submission_csv = os.path.join(output_dir, "submission.csv")
        submission_time_csv = os.path.join(output_dir, "submission_time.csv")

        logger.info("Loading input JSON: %s", input_path)
        samples_raw = load_private_json(input_path)
        samples = normalize_samples(samples_raw)

        if not samples:
            raise ValueError("No samples to run after normalization")
        logger.info("✓ Loaded %d samples", len(samples))

        # Validate knowledge base
        if not os.path.exists(KB_PATH):
            raise FileNotFoundError(f"Knowledge base not found: {KB_PATH}")

        logger.info("Initializing RAG system (BGE-M3 + Qwen-Rerank)...")
        rag = RAGSystem(kb_path=KB_PATH)
        logger.info("✓ RAG initialized")

        logger.info("Initializing vLLM (%s)...", MODEL_NAME)
        llm = LLM(
            model=MODEL_NAME,
            trust_remote_code=True,
            gpu_memory_utilization=GPU_MEMORY_UTIL,
            max_model_len=MAX_MODEL_LEN,
        )
        sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        logger.info("✓ vLLM ready")

        submission_rows: List[Dict[str, str]] = []
        submission_time_rows: List[Dict[str, Any]] = []

        logger.info("Running inference per-sample to measure time...")
        for qid, question in tqdm(samples, total=len(samples), desc="Infer"):
            context = rag.search(question, top_k=RAG_TOP_K)
            prompt = prepare_prompt(question=question, context=context)

            start = time.perf_counter()
            outputs = llm.generate([prompt], sampling_params)
            end = time.perf_counter()

            # outputs is a list with length 1
            generated_text = ""
            try:
                generated_text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
            except Exception:
                generated_text = ""

            answer = extract_answer(generated_text)
            elapsed = end - start

            submission_rows.append({"qid": qid, "answer": answer})
            submission_time_rows.append({"qid": qid, "answer": answer, "time": elapsed})

        pd.DataFrame(submission_rows).to_csv(submission_csv, index=False)
        pd.DataFrame(submission_time_rows).to_csv(submission_time_csv, index=False)

        logger.info("✓ Saved %s", submission_csv)
        logger.info("✓ Saved %s", submission_time_csv)
        logger.info("Done.")
        return True

    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

