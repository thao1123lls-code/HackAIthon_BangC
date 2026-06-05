import pandas as pd
import os
import re
from vllm import LLM, SamplingParams
from rag import RAGSystem # Cú pháp Import module tra cứu vừa viết

def extract_answer(text):
    match = re.search(r'Đáp án cuối cùng là:\s*([ABCD])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    fallback = re.findall(r'\b([ABCD])\b', text)
    return fallback[-1] if fallback else "A"

def main():
    # Tự động nhận diện môi trường: chạy trên máy tính (local) hay trong Docker (khi chấm điểm)
    input_path = "/data/public_test.csv" if os.path.exists("/data") else "data/public_test.csv"
    output_dir = "/output" if os.path.exists("/output") else "output"
    output_path = f"{output_dir}/pred.csv"

    # 1. Khởi tạo song song cả 2 mô hình
    rag = RAGSystem(kb_path="knowledge_base.txt") # BGE-m3 chạy trước
    
    # ĐÃ SỬA: Hạ xuống bản 0.5B để test trên máy cá nhân không bị tràn RAM
    # Khi nộp bài lên Docker Hub, em đổi số 0.5B này thành 7B
   llm = LLM(model="Qwen/Qwen3.5-7B-Chat", trust_remote_code=True)
    
    sampling_params = SamplingParams(temperature=0.1, max_tokens=256)
    df = pd.read_csv(input_path)
    
    prompts = []
    for index, row in df.iterrows():
        question = row['question']
        
        # 2. Dùng BGE-m3 tra cứu tài liệu liên quan đến câu hỏi này
        context = rag.search(question, top_k=2)
        
        # 3. Ép ngữ cảnh vào Prompt (Bí kíp chống ảo giác)
        prompt = f"""Bạn là một chuyên gia. Hãy đọc kỹ thông tin tham khảo sau đây:
[THÔNG TIN THAM KHẢO]: {context}

Dựa vào thông tin trên, hãy giải quyết câu hỏi trắc nghiệm sau:
{question}

Hãy suy luận ngắn gọn. BẮT BUỘC chốt lại bằng câu: "Đáp án cuối cùng là: [X]" (Thay [X] bằng A, B, C hoặc D)."""
        prompts.append(prompt)

    # 4. Để Qwen đọc prompt (đã bao gồm tài liệu) và đưa ra đáp án
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        final_answer = extract_answer(generated_text)
        results.append({"qid": df.iloc[i]['qid'], "answer": final_answer})

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Luồng RAG + LLM đã hoàn tất! File kết quả lưu tại: {output_path}")

if __name__ == "__main__":
    main()