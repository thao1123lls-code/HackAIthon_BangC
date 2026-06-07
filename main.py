import pandas as pd
import os
import re
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    logger.error(f"Failed to import vLLM: {e}")
    logger.error("This system requires GPU support. Install with: pip install vllm")
    sys.exit(1)

from rag import RAGSystem

def extract_answer(generated_text):
    """Extract answer from generated text with 3-layer fallback strategy"""
    if not generated_text or not isinstance(generated_text, str):
        return "A"
    
    # Layer 1: Primary pattern "Đáp án: X"
    match = re.search(r'Đáp án:\s*([ABCD])', generated_text, re.IGNORECASE)
    if match: 
        return match.group(1).upper()
    
    # Layer 2: Keywords "chọn" or "là"
    match_2 = re.search(r'(chọn|là)\s*([ABCD])', generated_text, re.IGNORECASE)
    if match_2: 
        return match_2.group(2).upper()

    # Layer 3: Last ABCD character
    matches = re.findall(r'\b([ABCD])\b', generated_text)
    if matches: 
        return matches[-1].upper()
    
    # Layer 4: Default fallback
    logger.warning(f"Could not extract answer, using default 'A'. Text: {generated_text[:100]}")
    return "A"

def validate_input_file(file_path):
    """Validate input CSV file exists and has correct format"""
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
    
    # Check for missing values
    if df['question'].isna().any():
        logger.warning(f"Found {df['question'].isna().sum()} rows with missing questions")
        df = df.dropna(subset=['question'])
        logger.info(f"Removed rows with missing questions. Remaining: {len(df)} rows")
    
    if len(df) == 0:
        raise ValueError("All rows have missing questions after cleanup")
    
    logger.info(f"Validated input file: {len(df)} questions found")
    return df

def validate_knowledge_base(kb_path):
    """Validate knowledge base file exists and is readable"""
    if not os.path.exists(kb_path):
        logger.error(f"Knowledge base not found: {kb_path}")
        raise FileNotFoundError(f"Knowledge base not found: {kb_path}")
    
    try:
        with open(kb_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if len(content.strip()) == 0:
            logger.warning(f"Knowledge base is empty: {kb_path}")
        else:
            logger.info(f"Knowledge base loaded: {len(content)} characters")
        
        return content
    except Exception as e:
        logger.error(f"Failed to read knowledge base {kb_path}: {e}")
        raise ValueError(f"Failed to read knowledge base: {e}")

def ensure_output_directory(output_dir):
    """Ensure output directory exists and is writable"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Test write permission
        test_file = os.path.join(output_dir, ".test")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        
        logger.info(f"Output directory ready: {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to create/access output directory {output_dir}: {e}")
        raise PermissionError(f"Cannot write to output directory: {e}")

def main():
    """Main inference pipeline with comprehensive error handling"""
    try:
        # Step 1: Determine paths (Docker vs local)
        input_path = "/data/public_test.csv" if os.path.exists("/data") else "data/public_test.csv"
        output_dir = "/output" if os.path.exists("/output") else "output"
        output_path = os.path.join(output_dir, "pred.csv")
        kb_path = "knowledge_base.txt"
        
        logger.info(f"Input path: {input_path}")
        logger.info(f"Output path: {output_path}")
        logger.info(f"Knowledge base: {kb_path}")
        
        # Step 2: Validate all input files
        logger.info("Validating input files...")
        df = validate_input_file(input_path)
        validate_knowledge_base(kb_path)
        ensure_output_directory(output_dir)
        
        # Step 3: Initialize RAG system
        logger.info("Initializing RAG system...")
        try:
            rag = RAGSystem(kb_path=kb_path)
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
        
        # Step 4: Initialize vLLM
        logger.info("Initializing vLLM (loading model)...")
        try:
            llm = LLM(
                model="Qwen/Qwen3.5-7B-Chat", 
                trust_remote_code=True,
                gpu_memory_utilization=0.9,  # Use 90% VRAM for batch processing
                max_model_len=2048  # Limit context length for speed
            )
            logger.info("vLLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            logger.error("Ensure you have a compatible GPU with CUDA support")
            raise
        
        # Step 5: Prepare prompts
        logger.info(f"Preparing {len(df)} prompts...")
        sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
        prompts = []
        
        for index, row in df.iterrows():
            try:
                question = str(row['question']).strip()
                
                if not question:
                    logger.warning(f"Row {index}: Empty question, skipping")
                    continue
                
                # Get context from RAG
                context = rag.search(question, top_k=2)
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
            
            except Exception as e:
                logger.warning(f"Row {index}: Failed to prepare prompt - {e}")
                continue
        
        if not prompts:
            logger.error("No valid prompts to generate. Check input file.")
            raise ValueError("No valid prompts found in input file")
        
        logger.info(f"Prepared {len(prompts)} prompts (skipped {len(df) - len(prompts)})")
        
        # Step 6: Generate answers
        logger.info(f"Generating answers for {len(prompts)} questions...")
        try:
            outputs = llm.generate(prompts, sampling_params)
            logger.info(f"Generated {len(outputs)} answers successfully")
        except Exception as e:
            logger.error(f"Failed to generate answers: {e}")
            raise
        
        # Step 7: Extract and compile results
        logger.info("Extracting final answers...")
        results = []
        
        for i, output in enumerate(outputs):
            try:
                generated_text = output.outputs[0].text if output.outputs else ""
                final_answer = extract_answer(generated_text)
                qid = df.iloc[i]['qid']
                results.append({"qid": qid, "answer": final_answer})
            except Exception as e:
                logger.warning(f"Question {i}: Failed to extract answer - {e}, using default 'A'")
                results.append({"qid": df.iloc[i]['qid'], "answer": "A"})
        
        # Step 8: Save results
        logger.info(f"Saving {len(results)} results to {output_path}...")
        try:
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_path, index=False)
            logger.info(f"✓ Successfully saved predictions to {output_path}")
            logger.info(f"Result format: {len(results_df)} rows, columns: {list(results_df.columns)}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
        
        return True
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)