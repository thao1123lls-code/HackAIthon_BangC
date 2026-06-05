# Sử dụng base image Python chuẩn
FROM python:3.10-slim

# Cài đặt môi trường làm việc
WORKDIR /app

# Copy file thư viện vào và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tải sẵn trọng số mô hình vào trong image (cực kỳ quan trọng để chạy offline)
# Script này sẽ ép Python tải model Qwen1.5-0.5B-Chat về lưu sẵn
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('Qwen/Qwen1.5-0.5B-Chat'); \
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-0.5B-Chat')"

# Copy toàn bộ code vào trong container
COPY . .

# Khai báo các thư mục được mount từ bên ngoài
VOLUME ["/data", "/output"]

# Điểm neo bắt đầu chạy hệ thống
ENTRYPOINT ["python", "main.py"]