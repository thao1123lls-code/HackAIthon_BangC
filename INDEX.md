# 📖 Documentation Index

Chào mừng đến với **HackAIthon 2026 - Giải pháp Bảng C**! 

File này giúp bạn tìm tài liệu phù hợp với nhu cầu.

---

## 🎯 Bạn muốn làm gì?

### ⚡ "Tôi chỉ muốn chạy nó ngay bây giờ"
→ **[QUICK_START.md](QUICK_START.md)** (5 phút)
- Cài đặt nhanh với Docker
- Test hệ thống
- Run inference

### 🤔 "Tôi muốn hiểu hệ thống hoạt động như thế nào"
→ **[ARCHITECTURE.md](ARCHITECTURE.md)** (Chi tiết)
- Kiến trúc 4 tầng
- Data flow & timing
- Công thức toán học
- Memory optimization

### 💻 "Tôi cần API reference & hàm"
→ **[API_REFERENCE.md](API_REFERENCE.md)** (Reference)
- Class RAGSystem
- Function extract_answer()
- vLLM configuration
- Error handling
- Code examples

### ⚙️ "Tôi muốn tối ưu accuracy hoặc tốc độ"
→ **[TUNING_GUIDE.md](TUNING_GUIDE.md)** (Advanced)
- Cải thiện accuracy
- Tăng throughput
- Profiling & debugging
- Fine-tuning

### 📚 "Tôi muốn hiểu tất cả"
→ **[README.md](README.md)** (Đầy đủ)
- Tóm tắt dự án
- Công nghệ sử dụng
- Hướng dẫn cài đặt
- Hiệu suất

---

## 📁 Cấu trúc File

```
d:\HackAIthon_BangC\
├── 📖 Tài liệu
│   ├── README.md              # ← Bắt đầu từ đây
│   ├── QUICK_START.md         # Chạy nhanh
│   ├── ARCHITECTURE.md        # Chi tiết kiến trúc
│   ├── API_REFERENCE.md       # API reference
│   ├── TUNING_GUIDE.md        # Tối ưu hóa
│   └── INDEX.md               # File này (directory)
│
├── 💻 Code chính
│   ├── main.py                # Script inference chính
│   ├── rag.py                 # RAG system (2 chặng)
│   ├── requirements.txt        # Dependencies
│   └── Dockerfile             # Docker config
│
├── 📊 Data
│   ├── data/
│   │   └── public_test.csv     # Input: Câu hỏi test
│   ├── output/
│   │   └── pred.csv           # Output: Kết quả dự đoán
│   └── knowledge_base.txt      # Knowledge base
│
└── 📋 Config
    └── (Trống - mở rộng sau)
```

---

## 🚀 Learning Path (Suggested)

### Người mới
```
1. Đọc README.md - Hiểu tổng quát (5 min)
2. Chạy QUICK_START.md - Run hệ thống (5 min)
3. Kiểm tra kết quả output/pred.csv (1 min)
```
**Tổng thời gian: ~15 phút**

### Người muốn hiểu sâu
```
1. README.md - Overview (10 min)
2. ARCHITECTURE.md - Kiến trúc chi tiết (30 min)
3. API_REFERENCE.md - Code level (20 min)
4. TUNING_GUIDE.md - Tối ưu (20 min)
```
**Tổng thời gian: ~80 phút**

### Người muốn cải thiện performance
```
1. TUNING_GUIDE.md - Phần "Tối ưu Accuracy" (15 min)
2. ARCHITECTURE.md - Phần "Công thức tối ưu" (15 min)
3. Thực hành: Thay đổi parameters & test (30+ min)
```
**Tổng thời gian: ~60+ phút**

---

## 📋 Tóm tắt từng File

### 1. README.md
**Dài:** ~400 lines  
**Thời gian đọc:** 15-20 min

**Nội dung:**
- Tóm tắt dự án
- Kiến trúc hệ thống (diagram)
- 3 công nghệ chính (LLM, RAG, vLLM)
- Hướng dẫn cài đặt (Docker + Local)
- Hướng dẫn chạy
- Chi tiết từng component
- Công thức tối ưu
- Hiệu suất
- Troubleshooting

**Khi nào đọc?** Muốn hiểu tổng quát dự án

---

### 2. QUICK_START.md
**Dài:** ~300 lines  
**Thời gian đọc:** 5-10 min

**Nội dung:**
- Điều kiện tiên quyết
- 2 cách setup (Docker + Local)
- Test hệ thống
- Troubleshooting nhanh
- Tips & tricks

**Khi nào đọc?** Muốn chạy hệ thống ngay lập tức

---

### 3. ARCHITECTURE.md
**Dài:** ~700 lines  
**Thời gian đọc:** 40-50 min

**Nội dung:**
- Data flow chi tiết
- RAG System (2 chặng):
  - Chunking strategy
  - Vector retrieval
  - Semantic reranking
- LLM Engine (vLLM):
  - GPU optimization
  - KV cache
  - Temperature & greedy decoding
- Prompting (Reflective CoT)
- Answer extraction (3-layer)
- Timing breakdown
- Memory architecture
- Error handling

**Khi nào đọc?** Muốn hiểu sâu cách hoạt động

---

### 4. API_REFERENCE.md
**Dài:** ~500 lines  
**Thời gian đọc:** 30-40 min

**Nội dung:**
- RAGSystem class & methods
- extract_answer() function
- main() function
- vLLM configuration
- Data models (CSV format)
- Error handling
- Usage examples
- Custom implementations

**Khi nào đọc?** Muốn viết code với hệ thống

---

### 5. TUNING_GUIDE.md
**Dài:** ~600 lines  
**Thời gian đọc:** 35-45 min

**Nội dung:**
- Hiệu suất hiện tại
- Tối ưu Accuracy:
  - Cải thiện KB
  - Tuỳ chỉnh top-k
  - Prompt engineering
  - Reranker enabling
- Tối ưu Tốc độ:
  - Batch processing
  - Sequence length
  - GPU acceleration
  - Quantization
- Profiling tools
- Troubleshooting
- Fine-tuning
- Checklist

**Khi nào đọc?** Muốn cải thiện accuracy/tốc độ

---

## 🎓 Key Concepts

| Khái niệm | Giải thích | File |
|-----------|-----------|------|
| **RAG** | Retrieval Augmented Generation | ARCHITECTURE.md |
| **BGE-M3** | Vector embedding model | ARCHITECTURE.md §1 |
| **Qwen-Rerank** | Semantic reranker | ARCHITECTURE.md §2 |
| **vLLM** | Fast LLM inference | ARCHITECTURE.md §3 |
| **Reflective CoT** | Chain-of-thought with elimination | ARCHITECTURE.md §4 |
| **KV Cache** | GPU memory optimization | ARCHITECTURE.md §6 |
| **Chunking** | Text splitting with overlap | ARCHITECTURE.md §1 |

---

## 💡 Quick Reference

### Commands

```bash
# Setup
docker pull phuongthao12082007/phuongthao_innovator:v1
pip install -r requirements.txt

# Run
python main.py                    # Full inference
python rag.py                     # Test RAG only
python test_single.py             # Single question

# Monitor
nvidia-smi                        # Check GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Code Snippets

```python
# Inference một câu
from rag import RAGSystem
from vllm import LLM, SamplingParams

rag = RAGSystem("knowledge_base.txt")
llm = LLM("Qwen/Qwen3.5-7B-Chat")
context = rag.search("Câu hỏi")
output = llm.generate([prompt], SamplingParams(temperature=0.0))
```

### Troubleshooting

| Error | Solution |
|-------|----------|
| CUDA OOM | Giảm `gpu_memory_utilization` |
| Model not found | Tải manual với `transformers` |
| Slow inference | Giảm `max_model_len` |
| Low accuracy | Cải thiện KB hoặc prompt |

---

## 📞 Support

1. **Có lỗi?** → Check [Troubleshooting](#-quick-reference)
2. **Không hiểu?** → Đọc relevant doc ở [Learning Path](#-learning-path-suggested)
3. **Muốn tối ưu?** → [TUNING_GUIDE.md](TUNING_GUIDE.md)
4. **Cần code?** → [API_REFERENCE.md](API_REFERENCE.md)

---

## 📊 Statistics

| Metric | Giá trị |
|--------|--------|
| **Tổng documentation** | ~2000 lines |
| **Code** | ~250 lines (main.py + rag.py) |
| **Supported models** | Qwen3.5-7B, BGE-m3, Qwen-Rerank |
| **Min VRAM** | 10GB |
| **Max VRAM** | 16GB |
| **Latency/question** | 50-100ms |
| **Throughput** | 50-100 q/sec |
| **Accuracy** | 75-85% |

---

## ✅ Checklist Trước Khi Bắt Đầu

- [ ] Đọc README.md
- [ ] Cài đặt Docker hoặc Local
- [ ] Test GPU: `nvidia-smi`
- [ ] Run QUICK_START.md
- [ ] Check output/pred.csv
- [ ] (Optional) Đọc ARCHITECTURE.md
- [ ] (Optional) Tối ưu với TUNING_GUIDE.md

---

## 🎉 Chúc mừng!

Bạn đã sẵn sàng. Hãy chọn file phù hợp và bắt đầu! 🚀

```bash
python main.py  # Go!
```

---

**Last Updated:** 2026-06-07  
**Thí sinh:** Hoàng Phương Thảo  
**Loại thi:** Cá nhân  
**Cuộc thi:** HackAIthon 2026 - Bảng C
