# 🚀 Quick Start Guide

**Bạn muốn chạy hệ thống ngay bây giờ?** Hãy làm theo hướng dẫn này - chỉ 5 phút!

---

## 📋 Điều kiện tiên quyết

### Khuyên (Optimized)
- **GPU:** NVIDIA GPU với CUDA support (RTX 4090, A100, H100, v.v.)
  - VRAM: 12GB+ (14GB cho FP16)
  - Speed: 50-100ms per question
- **RAM:** 16GB+
- **Disk:** 50GB (cho models)

### Tối thiểu (CPU-only, chậm)
- **CPU:** 8+ cores
- **RAM:** 28GB+ (FP32 model)
- **Disk:** 50GB (cho models)
- **Speed:** 2-5 seconds per question (CPU)

> **Ghi chú:** Ban tổ chức không yêu cầu bắt buộc GPU, nhưng hệ thống này được tối ưu cho GPU để đạt hiệu suất tốt nhất (80%+ accuracy, 50ms latency).

---

## ✨ Cách 1: Chạy với Docker (Khuyên dùng)

Nhanh nhất, không cần cài đặt phức tạp.

### Bước 1: Cài Docker & Docker GPU Support

```bash
# Ubuntu
sudo apt-get install docker.io docker-compose
sudo apt-get install nvidia-docker2

# Windows
# Tải Docker Desktop: https://www.docker.com/products/docker-desktop
```

### Bước 2: Pull Image

```bash
docker pull phuongthao12082007/phuongthao_innovator:v1
```

### Bước 3: Chuẩn bị dữ liệu

```bash
# Tạo thư mục
mkdir -p /path/to/data /path/to/output

# Copy file dữ liệu
cp data/public_test.csv /path/to/data/
```

### Bước 4: Chạy Container

```bash
docker run --gpus all \
  -v /path/to/data:/data \
  -v /path/to/output:/output \
  phuongthao12082007/phuongthao_innovator:v1
```

**Output:**
```
Đang tải mô hình BGE-m3 (Chặng 1 - Truy xuất thô)...
Đang tải mô hình Qwen-Rerank (Chặng 2 - Lọc tinh)...
Hệ thống Advanced RAG Sẵn sàng!
Luồng Inference Tối ưu đã hoàn tất! Kết quả tại: /output/pred.csv
```

✅ Xong! Kết quả ở `/path/to/output/pred.csv`

---

## ⚡ Cách 2: Cài Local (Linux/WSL2)

### Bước 1: Clone Repository

```bash
cd /d/HackAIthon_BangC
```

### Bước 2: Tạo Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
.\venv\Scripts\activate   # Windows
```

### Bước 3: Cài Dependencies

```bash
pip install -r requirements.txt
```

**Nếu bị lỗi CUDA:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Bước 4: Kiểm tra GPU

```bash
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
# Output: GPU available: True
```

### Bước 5: Chạy Inference

```bash
python main.py
```

**Lần đầu tiên:**
- Sẽ tải models (~50GB) → Mất 5-10 phút
- Lần tiếp theo chỉ cần 1-2 phút

✅ Kết quả sẽ được lưu ở `output/pred.csv`

---

## 🧪 Test Hệ thống

### Test 1: Kiểm tra RAG

```bash
python rag.py
```

**Output:**
```
Đang tải mô hình BGE-m3 (Chặng 1 - Truy xuất thô)...
Đang tải mô hình Qwen-Rerank (Chặng 2 - Lọc tinh)...
Hệ thống Advanced RAG Sẵn sàng!

--- KẾT QUẢ TRA CỨU ---
[Kết quả tra cứu ở đây...]
```

✅ Nếu có output → RAG working!

### Test 2: Inference Một Câu

```python
# test_single.py
import pandas as pd
from main import extract_answer
from rag import RAGSystem
from vllm import LLM, SamplingParams

rag = RAGSystem("knowledge_base.txt")
llm = LLM("Qwen/Qwen3.5-7B-Chat", gpu_memory_utilization=0.9, max_model_len=2048)
sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

question = "Python là gì?"
context = rag.search(question, top_k=2)

prompt = f"""Bạn là một hệ thống AI xuất sắc trong việc thi trắc nghiệm.
[THÔNG TIN THAM KHẢO]:
{context}

Câu hỏi: {question}
A. Ngôn ngữ lập trình
B. Con rắn
C. Loại máy bay
D. Công ty

--- QUY TRÌNH SUY LUẬN BẮT BUỘC ---
1. Phân tích câu hỏi.
2. Đánh giá từng phương án A, B, C, D.
3. Loại trừ các phương án sai.
4. Chốt đáp án.

Dòng cuối cùng BẮT BUỘC có định dạng: "Đáp án: X"
"""

output = llm.generate([prompt], sampling_params)[0].outputs[0].text
answer = extract_answer(output)

print(f"Question: {question}")
print(f"LLM Output:\n{output}")
print(f"Extracted Answer: {answer}")
```

```bash
python test_single.py
```

✅ Nếu in ra đáp án (A/B/C/D) → LLM working!

---

## 📊 Dữ liệu Input/Output

### Input: `data/public_test.csv`

```csv
qid,question,option_a,option_b,option_c,option_d
1,Python là ngôn ngữ lập trình bậc cao. Đặc điểm nào là ĐÚNG?,Cú pháp phức tạp,Cú pháp đơn giản và dễ học,Chỉ dùng cho web,Không có thư viện
```

### Output: `output/pred.csv`

```csv
qid,answer
1,B
```

---

## ⚠️ Troubleshooting

### ❌ GPU Memory Error

```
RuntimeError: CUDA out of memory
```

**Fix:**

```python
# main.py - Giảm gpu_memory_utilization từ 0.9 → 0.7
llm = LLM(
    model="Qwen/Qwen3.5-7B-Chat",
    gpu_memory_utilization=0.7,  # ← Đổi này
    max_model_len=2048
)
```

### ❌ Model Not Found

```
FileNotFoundError: Qwen/Qwen3.5-7B-Chat not found
```

**Fix:**

```bash
# Tải model manually
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen3.5-7B-Chat')"
```

### ❌ CUDA Not Available

```python
import torch
print(torch.cuda.is_available())  # False
```

**Fix:**

```bash
# Cài PyTorch CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### ❌ Slow Inference (> 200ms/câu)

**Diagnosis:**
```bash
nvidia-smi  # Check GPU utilization
```

**Possible Causes:**
- GPU bộ nhớ không đủ → Giảm `gpu_memory_utilization`
- KB quá lớn → Giảm `top_k` từ 2 → 1
- Batch size nhỏ → (vLLM tự optimize, không cần lo)

---

## 📈 Hiệu suất Mong đợi

| Metric | Giá trị |
|--------|--------|
| **Latency/câu** | 50-100ms |
| **Throughput** | 50-100 questions/second |
| **Accuracy** | 75-85% (tùy KB) |
| **VRAM** | 10-14GB |
| **Memory (CPU)** | 4-6GB |

**Ví dụ:** 1000 câu hỏi → ~10-20 giây (on 1 GPU)

---

## 🎓 Tiếp Theo: Tìm hiểu thêm

| Chủ đề | File |
|--------|------|
| **Kiến trúc chi tiết** | [ARCHITECTURE.md](ARCHITECTURE.md) |
| **Tối ưu hóa** | [TUNING_GUIDE.md](TUNING_GUIDE.md) |
| **API Reference** | [API_REFERENCE.md](API_REFERENCE.md) |
| **README đầy đủ** | [README.md](README.md) |

---

## 💡 Tips & Tricks

### Tip 1: Tăng Throughput với Batch Processing

```python
# Thay vì xử lý 1 câu hỏi lúc một
# Xử lý tất cả cùng lúc
prompts = [build_prompt(q) for q in all_questions]
outputs = llm.generate(prompts, sampling_params)  # Batch inference
```

### Tip 2: Kiểm tra GPU Memory

```bash
# Real-time monitoring
watch -n 0.5 nvidia-smi
```

### Tip 3: Tìm hiểu Prompt Engineering

Thay đổi prompt template trong `main.py`:

```python
# Thêm few-shot examples
prompt = f"""
Ví dụ 1:
Câu: "Là gì?"
→ Đáp án: A

Ví dụ 2:
Câu: "Là gì?"
→ Đáp án: B

Bây giờ:
{question}
"""
```

### Tip 4: Cache Models Locally

```bash
# Models sẽ tự cache ở ~/.cache/huggingface/
# Lần chạy tiếp theo sẽ nhanh hơn (không cần download)
```

---

## 🐛 Báo Lỗi

Nếu có vấn đề, kiểm tra:

1. **GPU working?** `nvidia-smi`
2. **CUDA installed?** `python -c "import torch; print(torch.cuda.is_available())"`
3. **Models downloaded?** `~/.cache/huggingface/`
4. **Input file format correct?** Check `data/public_test.csv`
5. **Output directory writable?** Check folder permissions

---

## ✅ Checklist Trước Khi Deploy

- [ ] GPU test passed
- [ ] CUDA 11.8+ installed
- [ ] Models downloaded (~50GB)
- [ ] Input CSV format correct
- [ ] Output directory writable
- [ ] Run test_single.py successfully
- [ ] No CUDA memory errors
- [ ] Latency < 200ms/question
- [ ] Accuracy > 70%

---

## 🎉 Chúc mừng!

Bạn đã sẵn sàng! 🚀

```bash
python main.py  # Đáp án của bạn sẽ ở output/pred.csv
```

---

## 📞 Cần Giúp Đỡ?

1. Kiểm tra [Troubleshooting](#-troubleshooting)
2. Đọc [ARCHITECTURE.md](ARCHITECTURE.md) để hiểu chi tiết
3. Tham khảo [API_REFERENCE.md](API_REFERENCE.md)
4. Xem [TUNING_GUIDE.md](TUNING_GUIDE.md) để tối ưu hóa

Happy coding! 🎓
