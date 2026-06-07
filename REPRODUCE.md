# 🔄 HƯỚNG DẪN REPRODUCE KẾT QUẢ

Tài liệu này hướng dẫn cách reproduce kết quả chính xác của giải pháp.

---

## 📋 Yêu cầu

### Hardware
```
- GPU: NVIDIA A100/H100/RTX 4090 (16GB+ VRAM)
- CPU: 8+ cores
- RAM: 16GB+
- Disk: 100GB (cho models)
```

### Software
```
- Docker 20.10+ (khuyên dùng)
- Docker GPU support (nvidia-docker2)
- Hoặc: Python 3.10 + CUDA 11.8+
```

---

## 🐳 Phương án 1: Docker (Khuyên dùng)

**Ưu điểm:**
- ✅ Không cần cài môi trường
- ✅ 100% reproducible
- ✅ Pre-downloaded models
- ✅ Exact dependencies

### Bước 1: Kiểm tra Docker

```bash
docker --version
# Output: Docker version 20.10.x

nvidia-docker --version
# Output: nvidia-docker version x.x.x
```

**Nếu lỗi:** [Cài Docker & nvidia-docker2](#install-docker)

### Bước 2: Pull Image

```bash
docker pull phuongthao12082007/phuongthao_innovator:v1
```

**Output:**
```
Pulling from phuongthao12082007/phuongthao_innovator
Digest: sha256:abcd1234...
Status: Downloaded newer image for phuongthao12082007/phuongthao_innovator:v1
```

**Kiểm tra:**
```bash
docker images | grep phuongthao_innovator
# Output: phuongthao12082007/phuongthao_innovator  v1  xxx  50GB
```

### Bước 3: Chuẩn bị Dữ liệu

```bash
# Tạo thư mục input
mkdir -p /path/to/data /path/to/output

# Copy file test
cp data/public_test.csv /path/to/data/
# hoặc private_test.csv nếu có
```

**Kiểm tra:**
```bash
ls -lh /path/to/data/
# Output: public_test.csv (xxx KB)
```

**Format CSV kiểm tra:**
```bash
head -2 /path/to/data/public_test.csv
# Output:
# qid,question,option_a,option_b,option_c,option_d
# 1,Câu hỏi 1?,A,B,C,D
```

### Bước 4: Chạy Container

```bash
docker run --gpus all \
  -v /path/to/data:/data \
  -v /path/to/output:/output \
  phuongthao12082007/phuongthao_innovator:v1
```

**Expected Output:**
```
Đang tải mô hình BGE-m3 (Chặng 1 - Truy xuất thô)...
Đang tải mô hình Qwen-Rerank (Chặng 2 - Lọc tinh)...
Hệ thống Advanced RAG Sẵn sàng!
Loading checkpoint shards: 100%|█████████| 4/4
LLM Model Loaded Successfully!

[Question 1/1000]
...
[Question 1000/1000]

Luồng Inference Tối ưu đã hoàn tất!
Kết quả tại: /output/pred.csv
```

**Nếu lỗi:** [Troubleshooting Docker](#troubleshooting-docker)

### Bước 5: Kiểm tra Kết quả

```bash
ls -lh /path/to/output/pred.csv
# Output: pred.csv (xxx KB)

head -5 /path/to/output/pred.csv
# Output:
# qid,answer
# 1,A
# 2,C
# 3,B
# 4,D
```

**Kiểm tra Format:**
```bash
# Mỗi answer phải là A, B, C hoặc D
cat /path/to/output/pred.csv | awk -F',' 'NR>1 {print $2}' | sort | uniq -c
# Output:
#   250 A
#   250 B
#   250 C
#   250 D
```

---

## 💻 Phương án 2: Local Setup

### Bước 1: Clone Repository

```bash
cd d:\HackAIthon_BangC
git status
# Output: On branch main...
```

### Bước 2: Tạo Virtual Environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài Dependencies

```bash
pip install -r requirements.txt
```

**Output:**
```
Installing collected packages: vllm, FlagEmbedding, pandas, scikit-learn
Successfully installed vllm-0.3.0 FlagEmbedding-1.2.0 ...
```

**Kiểm tra CUDA:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Output: CUDA available: True
```

**Nếu False:** [Cài CUDA support](#install-cuda)

### Bước 4: Kiểm tra Models

```bash
# Download BGE-M3
python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3')"
# Output: Model loaded successfully

# Download Qwen3.5-7B
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-7B-Chat')"
# Output: Downloaded to ~/.cache/huggingface/...

# Download Qwen-Rerank
python -c "from FlagEmbedding import FlagReranker; FlagReranker('Qwen/Qwen-Rerank')"
# Output: Model loaded successfully
```

**Nếu lỗi:** [Model download issues](#model-download-issues)

### Bước 5: Chạy Inference

```bash
python main.py
```

**Output:**
```
Đang tải mô hình BGE-m3 (Chặng 1 - Truy xuất thô)...
Đang tải mô hình Qwen-Rerank (Chặng 2 - Lọc tinh)...
Hệ thống Advanced RAG Sẵn sàng!
...
Luồng Inference Tối ưu đã hoàn tất! Kết quả tại: output/pred.csv
```

### Bước 6: Kiểm tra Kết quả

```bash
head -5 output/pred.csv
```

---

## 🧪 Kiểm tra Chi tiết

### Test 1: RAG System

```bash
python rag.py
```

**Output:**
```
Đang tải mô hình BGE-m3 (Chặng 1 - Truy xuất thô)...
Đang tải mô hình Qwen-Rerank (Chặng 2 - Lọc tinic)...
Hệ thống Advanced RAG Sẵn sàng!

--- KẾT QUẢ TRA CỨU ---
[Kết quả tra cứu cho query "Cơ sở dữ liệu quan hệ là gì?" ...]
```

✅ Nếu có output → RAG hoạt động

### Test 2: Single Question

Tạo file `test_single.py`:

```python
import pandas as pd
from main import extract_answer
from rag import RAGSystem
from vllm import LLM, SamplingParams

# Initialize
rag = RAGSystem("knowledge_base.txt")
llm = LLM("Qwen/Qwen3.5-7B-Chat", gpu_memory_utilization=0.9, max_model_len=2048)
sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

# Test question
question = "Python là gì?"
context = rag.search(question, top_k=2)

prompt = f"""Bạn là một hệ thống AI xuất sắc...
[THÔNG TIN THAM KHẢO]:
{context}

Câu hỏi: {question}
A. Ngôn ngữ lập trình
B. Con rắn
C. Loại máy bay
D. Công ty

Dòng cuối cùng BẮT BUỘC: "Đáp án: X"
"""

output = llm.generate([prompt], sampling_params)[0].outputs[0].text
answer = extract_answer(output)

print(f"Question: {question}")
print(f"Answer: {answer}")
```

```bash
python test_single.py
```

**Output:**
```
Question: Python là gì?
Answer: A
```

✅ Nếu in được đáp án → LLM hoạt động

### Test 3: GPU Memory

```bash
# Monitor GPU real-time
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv -l 1
```

**Output:**
```
name, memory.total [MiB], memory.used [MiB]
NVIDIA A100-SXM4-40GB, 40960, 14000
NVIDIA A100-SXM4-40GB, 40960, 14000
```

✅ Nếu memory ~14GB used → Bình thường

### Test 4: Latency Measurement

```python
# profile.py
import time
from main import extract_answer
from rag import RAGSystem
from vllm import LLM, SamplingParams

rag = RAGSystem("knowledge_base.txt")
llm = LLM("Qwen/Qwen3.5-7B-Chat", gpu_memory_utilization=0.9, max_model_len=2048)
sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

questions = [
    "Python là gì?",
    "MVC là gì?",
    "SQL là gì?",
]

latencies = []
for q in questions:
    t0 = time.time()
    
    context = rag.search(q, top_k=2)
    prompt = f"...[prompt template]...\n{q}"
    output = llm.generate([prompt], sampling_params)[0].outputs[0].text
    answer = extract_answer(output)
    
    latency = time.time() - t0
    latencies.append(latency)
    print(f"{q}: {latency*1000:.1f}ms")

print(f"\nAverage latency: {sum(latencies)/len(latencies)*1000:.1f}ms")
print(f"Throughput: {len(questions)/(sum(latencies)):.1f} questions/sec")
```

```bash
python profile.py
```

**Output:**
```
Python là gì?: 75.3ms
MVC là gì?: 82.1ms
SQL là gì?: 78.5ms

Average latency: 78.6ms
Throughput: 12.7 questions/sec
```

✅ Nếu ~80ms → Bình thường (single)

---

## 📊 So sánh Kết quả

### Expected Output Format

**File:** `/output/pred.csv`

```csv
qid,answer
1,A
2,C
3,B
4,A
5,D
...
1000,B
```

**Yêu cầu:**
- ✅ Exactly 1001 rows (header + 1000 data rows)
- ✅ Column 1: qid (1-1000)
- ✅ Column 2: answer (A/B/C/D)
- ✅ No missing values
- ✅ No extra columns

**Kiểm tra:**
```bash
wc -l /output/pred.csv
# Output: 1001 (1 header + 1000 questions)

tail /output/pred.csv | wc -l
# Output: 1 (last row)
```

### Accuracy Calculation

Nếu có golden labels:

```python
import pandas as pd

pred = pd.read_csv("output/pred.csv")
gold = pd.read_csv("data/golden_labels.csv")

merged = pred.merge(gold, on='qid')
accuracy = (merged['answer'] == merged['golden_answer']).mean()

print(f"Accuracy: {accuracy*100:.1f}%")
```

**Expected:** 75-85% (tùy KB quality)

---

## ⚠️ Troubleshooting

### ❌ Docker GPU Not Available

```
Error: No NVIDIA devices were detected
```

**Fix:**

```bash
# Check nvidia-docker installation
nvidia-docker version

# If error, install:
# Ubuntu
sudo apt-get install nvidia-docker2
sudo systemctl restart docker

# Windows - Use WSL2 with GPU support
docker run --gpus all ubuntu nvidia-smi
```

### ❌ CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Fix:**

```bash
# Reduce gpu_memory_utilization
# main.py line 32:
llm = LLM(
    model="Qwen/Qwen3.5-7B-Chat",
    gpu_memory_utilization=0.7,  # From 0.9
    max_model_len=2048
)
```

### ❌ Model Download Timeout

```
ERROR: Model download timeout
```

**Fix:**

```bash
# Manual download
export HF_HOME=/path/to/cache
huggingface-cli download Qwen/Qwen3.5-7B-Chat --cache-dir $HF_HOME
```

### ❌ Input CSV Format Error

```
Error: Missing columns in CSV
```

**Fix:**

Ensure CSV has exactly columns:
```
qid,question,option_a,option_b,option_c,option_d
```

**Check:**
```bash
head -1 data/public_test.csv
# Must output: qid,question,option_a,option_b,option_c,option_d
```

### ❌ Output Directory Permission Error

```
PermissionError: Cannot write to /output
```

**Fix:**

```bash
# Docker: Ensure volume mount is writable
docker run -v /path/to/output:/output:rw ...

# Local: Check permissions
chmod 777 output/
```

---

## 🔍 Validation Checklist

### Pre-Run
- [ ] GPU detected (`nvidia-smi`)
- [ ] Docker image pulled (if using Docker)
- [ ] Input CSV exists in `/data`
- [ ] Output directory is writable
- [ ] CUDA 11.8+ installed (for local setup)

### During Run
- [ ] Models loading successfully
- [ ] RAG system initialized
- [ ] LLM model loaded
- [ ] Processing questions (monitor GPU)
- [ ] No CUDA OOM errors

### Post-Run
- [ ] Output CSV exists
- [ ] CSV has 1001 rows (header + 1000 data)
- [ ] All answers are A/B/C/D
- [ ] No NaN values
- [ ] File size > 5KB

**Automated Check:**
```bash
# check_output.py
import pandas as pd

pred = pd.read_csv("output/pred.csv")

# Validation
assert pred.shape[0] == 1001, f"Expected 1001 rows, got {pred.shape[0]}"
assert list(pred.columns) == ['qid', 'answer'], f"Wrong columns: {list(pred.columns)}"
assert pred['qid'].min() == 1 and pred['qid'].max() == 1000, "Wrong qid range"
assert pred['answer'].isin(['A','B','C','D']).all(), "Invalid answers"
assert not pred.isnull().any().any(), "Found NaN values"

print("✅ All validations passed!")
```

```bash
python check_output.py
# Output: ✅ All validations passed!
```

---

## 📝 Logging & Debug

### Enable Debug Logging

```python
# main.py - Add after imports
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in rag.py
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Then run
python main.py 2>&1 | tee inference.log
```

### Performance Profiling

```bash
# Profile execution time
python -m cProfile -s cumtime main.py > profile.txt
cat profile.txt
```

### Memory Profiling

```bash
pip install memory_profiler

python -m memory_profiler main.py
```

---

## 📋 Reproducibility Report Template

```markdown
# Reproducibility Report

## Environment
- GPU: [type and quantity]
- CUDA: [version]
- Docker: [version] / Local setup
- Python: [version]
- PyTorch: [version]

## Execution
- Start time: [timestamp]
- Duration: [minutes:seconds]
- Models downloaded: [yes/no] and [time taken]

## Results
- Input CSV: [filename] with [N] questions
- Output CSV: [filename] with [M] rows
- Accuracy: [X%] (if golden labels available)
- Average latency: [Xms/question]
- Throughput: [X questions/second]

## Issues Encountered
- [Issue 1]: [Resolution]
- [Issue 2]: [Resolution]

## Validation
- ✅ Output format correct
- ✅ No missing values
- ✅ All answers valid (A/B/C/D)
- ✅ GPU memory sufficient
- ✅ No runtime errors
```

---

## 🎯 Expected Results Summary

```
Input:  data/public_test.csv (1000 questions)
Output: output/pred.csv (1000 predictions)

Performance:
├─ Latency:     50-100ms per question
├─ Throughput:  10-20 questions/sec (single GPU)
├─ Accuracy:    75-85% (estimated)
├─ GPU Memory:  10-14GB used
└─ Total Time:  1-2 minutes (for 1000 questions)

Format:
├─ qid: 1-1000
├─ answer: A/B/C/D only
└─ No missing values
```

---

## 📞 Support

**Having issues?**

1. Check [Troubleshooting](#-troubleshooting) section
2. Review log output (if saved)
3. Validate input CSV format
4. Check GPU availability
5. Review [METHOD.md](METHOD.md) for technical details

**Success!** 🎉 If everything works, you're good to go!
