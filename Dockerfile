# 1. Khai báo hệ điều hành nền
FROM python:3.10-slim

# 2. Thiết lập thư mục làm việc trong container
WORKDIR /app

# CHỮA LỖI: Cài đặt bộ công cụ biên dịch C++ (gcc, g++) để build vLLM
RUN apt-get update && apt-get install -y gcc g++ git build-essential

# 3. Copy file môi trường và cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Tải trước mô hình tra cứu BGE-m3
RUN python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3')"

# 5. Tải trước mô hình Qwen 7B 
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen1.5-7B-Chat')"

# 6. Copy toàn bộ mã nguồn vào container
COPY . .

# 7. Lệnh khởi chạy mặc định khi container được gọi
CMD ["python", "main.py"]