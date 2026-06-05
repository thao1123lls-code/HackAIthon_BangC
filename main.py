import pandas as pd
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_answer(text):
    # Dùng Regex để tìm ký tự A, B, C hoặc D trong câu trả lời của AI
    match = re.search(r'\b([ABCD])\b', text)
    return match.group(1) if match else "A" # Trả về A nếu AI trả lời lan man không rõ ý

def main():
    # 1. Định nghĩa đường dẫn I/O theo đúng chuẩn Ban tổ chức
    input_path = "/data/public_test.csv"
    output_path = "/output/pred.csv"

    # Tạo file test giả lập nếu chạy ở local để code không bị lỗi
    if not os.path.exists(input_path):
        os.makedirs("/data", exist_ok=True)
        pd.DataFrame([
            {"qid": "1", "question": "Hà Nội là thủ đô nước nào? A. Lào B. Việt Nam C. Thái Lan D. Mỹ"}
        ]).to_csv(input_path, index=False)

    # 2. Tải mô hình (sử dụng Qwen3.5 theo quy định)
    model_name = "Qwen/Qwen1.5-0.5B-Chat" # Dùng bản nhỏ gọn để test code I/O
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    # 3. Đọc dữ liệu
    df = pd.read_csv(input_path)
    results = []

    # 4. Đưa từng câu hỏi qua AI xử lý
    for index, row in df.iterrows():
        prompt = f"Bạn là một chuyên gia giải trắc nghiệm. Hãy chọn 1 đáp án đúng (A, B, C, D) cho câu hỏi sau và chỉ in ra chữ cái đó, không giải thích: {row['question']}"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=10)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        final_answer = extract_answer(response)
        results.append({"qid": row['qid'], "answer": final_answer})

    # 5. Ghi kết quả ra file chuẩn
    os.makedirs("/output", exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print("Đã hoàn thành và ghi file pred.csv thành công!")

if __name__ == "__main__":
    main()