"""
HackAIthon 2026 Bảng C - AI Multiple Choice QA System
Main inference engine with RAG + vLLM optimization
"""
from typing import Optional, Dict, List, Tuple
import pandas as pd
import os
import re
import sys
import logging
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    print(f"Failed to import vLLM: {e}")
    print("This system requires GPU support. Install with: pip install vllm")
    sys.exit(1)

from rag import RAGSystem

# ==================== CONSTANTS ====================
MODEL_NAME = "Qwen/Qwen3.5-7B-Chat"
GPU_MEMORY_UTIL = 0.9
MAX_MODEL_LEN = 2048
MAX_TOKENS = 256
TEMPERATURE = 0.0
DEFAULT_ANSWER = "A"
KB_PATH = "knowledge_base.txt"
RAG_TOP_K = 2
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# Pre-compile regex patterns for performance
REGEX_PATTERNS = {
    'primary': re.compile(r'Đáp án:\s*([ABCD])', re.IGNORECASE),
    'keywords': re.compile(r'(chọn|là)\s*([ABCD])', re.IGNORECASE),
    'last_choice': re.compile(r'\b([ABCD])\b'),
}

# Configure logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def extract_answer(generated_text: str) -> str:
    """
    Extract answer from generated text with 3-layer fallback strategy.
    
    Args:
        generated_text: LLM output text
    
    Returns:
        Single character: A, B, C, or D
    
    Performance: O(n) where n is text length, optimized with pre-compiled regex
    """
    if not generated_text or not isinstance(generated_text, str):
        return DEFAULT_ANSWER
    
    # Layer 1: Primary pattern "Đáp án: X" (fastest)
    match = REGEX_PATTERNS['primary'].search(generated_text)
    if match: 
        return match.group(1).upper()
    
    # Layer 2: Keywords "chọn" or "là" (medium)
    match_2 = REGEX_PATTERNS['keywords'].search(generated_text)
    if match_2: 
        return match_2.group(2).upper()

    # Layer 3: Last ABCD character (slower but comprehensive)
    matches = REGEX_PATTERNS['last_choice'].findall(generated_text)
    if matches: 
        return matches[-1].upper()
    
    # Layer 4: Default fallback
    logger.warning(f"Could not extract answer, using default '{DEFAULT_ANSWER}'. Text: {generated_text[:100]}")
    return DEFAULT_ANSWER

def validate_input_file(file_path: str) -> pd.DataFrame:
    """
    Validate input CSV file format and content.
    
    Args:
        file_path: Path to CSV file
    
    Returns:
        Cleaned DataFrame with validated questions
    
    Raises:
        FileNotFoundError, ValueError
    """
    if not os.path.exists(file_path):
        logger.error(f"Input file not found: {file_path}")
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Failed to read CSV file {file_path}: {e}")
        raise ValueError(f"Failed to parse CSV file: {e}")
    
    # Validate required columns
    required_cols = {'qid', 'question'}
    if not required_cols.issubset(df.columns):
        logger.error(f"CSV missing required columns. Required: {required_cols}, Found: {set(df.columns)}")
        raise ValueError(f"CSV must have columns: {required_cols}")
    
    # Check for empty dataframe
    if len(df) == 0:
        logger.error("Input CSV is empty")
        raise ValueError("Input CSV contains no data rows")
    
    # Clean missing values
    initial_len = len(df)
    df = df.dropna(subset=['question'])
    if len(df) < initial_len:
        logger.warning(f"Removed {initial_len - len(df)} rows with missing questions. Remaining: {len(df)}")
    
    if len(df) == 0:
        raise ValueError("All rows have missing questions after cleanup")
    
    logger.info(f"✓ Validated input: {len(df)} questions")
    return df

def validate_knowledge_base(kb_path: str) -> bool:
    """Validate knowledge base file exists and is readable."""
    if not os.path.exists(kb_path):
        logger.error(f"Knowledge base not found: {kb_path}")
        raise FileNotFoundError(f"Knowledge base not found: {kb_path}")
    
    try:
        with open(kb_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if len(content.strip()) == 0:
            logger.warning(f"Knowledge base is empty: {kb_path}")
        else:
            logger.info(f"✓ Knowledge base loaded: {len(content)} chars")
        return True
    except Exception as e:
        logger.error(f"Failed to read knowledge base: {e}")
        raise ValueError(f"Failed to read knowledge base: {e}")

def ensure_output_directory(output_dir: str) -> bool:
    """Ensure output directory exists and is writable."""
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Test write permission
        test_file = os.path.join(output_dir, ".test")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_file)
        return True
    except Exception as e:
        logger.error(f"Failed to ensure output directory: {e}")
        raise


def prepare_prompts(df: pd.DataFrame, rag: RAGSystem) -> Tuple[List[str], pd.DataFrame]:
    """
    Prepare prompts from questions with RAG context.
    
    Args:
        df: DataFrame with 'qid' and 'question' columns
        rag: Initialized RAG system
    
    Returns:
        Tuple of (prompts list, filtered dataframe)
    """
    prompts = []
    valid_indices = []
    
    for idx, (index, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Preparing prompts")):
        try:
            question = str(row['question']).strip()
            if not question:
                logger.warning(f"Row {index}: Empty question, skipping")
                continue
            
            # Get context from RAG
            context = rag.search(question, top_k=RAG_TOP_K)
            context_block = f"[THÔNG TIN THAM KHẢO]:\n{context}\n\n" if context else ""
            
            # Build prompt with reflective CoT
            prompt = f"""Bạn là một hệ thống AI xuất sắc trong việc thi trắc nghiệm.
{context_block}
Nhiệm vụ của bạn là giải quyết câu hỏi dưới đây một cách logic và cẩn thận:
{question}

--- QUY TRÌNH SUY LUẬN BẮT BUỘC ---
1. Phân tích ngữ cảnh và yêu cầu của câu hỏi.
2. Đánh giá ngắn gọn từng phương án A, B, C, D. Loại trừ các phương án sai.
3. Chốt lại đáp án đúng nhất.
Dòng cuối cùng BẮT BUỘC có định dạng: "Đáp án: X" (trong đó X là A, B, C hoặc D).
Không in thêm bất kỳ ký tự nào sau dòng đáp án."""
            
            prompts.append(prompt)
            valid_indices.append(index)
        
        except Exception as e:
            logger.warning(f"Row {index}: Failed to prepare - {e}")
            continue
    
    logger.info(f"✓ Prepared {len(prompts)} prompts (skipped {len(df) - len(prompts)})")
    return prompts, df.loc[valid_indices].reset_index(drop=True)

def main() -> bool:
    """
    Main inference pipeline with optimizations.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Step 1: Determine paths (Docker vs local)
        input_base = "/data" if os.path.exists("/data") else "data"
        output_dir = "/output" if os.path.exists("/output") else "output"
        output_path = os.path.join(output_dir, "pred.csv")

        # Prefer public_test.csv, fallback to private_test.csv (helps with more organizers)
        public_csv = os.path.join(input_base, "public_test.csv")
        private_csv = os.path.join(input_base, "private_test.csv")
        if os.path.exists(public_csv):
            input_path = public_csv
        elif os.path.exists(private_csv):
            input_path = private_csv
        else:
            # Keep original fallback for compatibility
            input_path = os.path.join(input_base, "public_test.csv")

        
        logger.info(f"Input: {input_path} | Output: {output_path}")
        
        # Step 2: Validate all input files
        logger.info("Step 1/6: Validating files...")
        df = validate_input_file(input_path)
        validate_knowledge_base(KB_PATH)
        ensure_output_directory(output_dir)
        
        # Step 3: Initialize RAG system
        logger.info("Step 2/6: Initializing RAG system...")
        rag = RAGSystem(kb_path=KB_PATH)
        
        # Step 4: Prepare prompts with progress bar
        logger.info("Step 3/6: Preparing prompts with RAG context...")
        prompts, df_valid = prepare_prompts(df, rag)
        
        if not prompts:
            raise ValueError("No valid prompts generated")
        
        # Step 5: Initialize vLLM
        logger.info("Step 4/6: Initializing vLLM (this may take 1-2 minutes)...")
        llm = LLM(
            model=MODEL_NAME, 
            trust_remote_code=True,
            gpu_memory_utilization=GPU_MEMORY_UTIL,
            max_model_len=MAX_MODEL_LEN
        )
        logger.info(f"✓ vLLM initialized with {MODEL_NAME}")
        
        # Step 6: Generate answers
        logger.info(f"Step 5/6: Generating answers ({len(prompts)} questions)...")
        sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        outputs = llm.generate(prompts, sampling_params)
        logger.info(f"✓ Generated {len(outputs)} answers")
        
        # Step 7: Extract and compile results
        logger.info("Step 6/6: Extracting final answers...")
        results = []
        
        for i, output in enumerate(tqdm(outputs, desc="Extracting answers")):
            try:
                generated_text = output.outputs[0].text if output.outputs else ""
                final_answer = extract_answer(generated_text)
                qid = df_valid.iloc[i]['qid']
                results.append({"qid": qid, "answer": final_answer})
            except Exception as e:
                logger.warning(f"Q{i}: Failed to extract - {e}, using '{DEFAULT_ANSWER}'")
                results.append({"qid": df_valid.iloc[i]['qid'], "answer": DEFAULT_ANSWER})
        
        # Step 8: Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        
        logger.info(f"✓ Successfully saved {len(results_df)} predictions")
        logger.info(f"✓ Output: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)