# Giải pháp Bảng C - HackAIthon 2026 (Đội: Innovator)

## Thông tin thí sinh
- **Họ và tên:** Hoàng Phương Thảo

## Kiến trúc Hệ thống (Cập nhật theo Thể lệ Vòng 2)
Hệ thống được thiết kế tối ưu tốc độ và độ chính xác tuyệt đối để xử lý bộ 2000 câu hỏi ẩn, tuân thủ đúng giới hạn mô hình của Ban tổ chức:

- **Bộ não (LLM):** `Qwen/Qwen3.5-7B-Chat` (Tối đa hóa khả năng suy luận ngữ cảnh Tiếng Việt).
- **Động cơ Suy luận (Inference Engine):** Sử dụng framework `vLLM` để ép xung tốc độ sinh văn bản, xử lý batch lớn mượt mà và tối ưu hóa VRAM.
- **Hệ thống Advanced RAG (Truy xuất 2 chặng):**
  - **Chặng 1 (Retrieval):** Sử dụng mô hình `BAAI/bge-m3` quét toàn bộ cơ sở dữ liệu để lấy Top 10 ngữ cảnh thô với tốc độ cực nhanh.
  - **Chặng 2 (Rerank):** Sử dụng `Qwen/Qwen-Rerank` làm giám khảo tái thẩm định sự logic giữa Câu hỏi và Ngữ cảnh, gạn lọc ra Top 2 đoạn văn tinh túy nhất để mớm cho LLM. Kỹ thuật này giúp triệt tiêu tối đa hiện tượng AI bị ảo giác (hallucination).
- **Kỹ thuật Prompting:** Áp dụng Chain-of-Thought (CoT) bắt buộc AI suy luận từng bước, kết hợp Regular Expression (Regex) ở khâu hậu xử lý để trích xuất chuẩn xác đáp án A, B, C hoặc D.

## Hướng dẫn Chấm điểm (Reproduce)

### 1. Đóng gói Docker Image
Hệ thống đã được thiết lập sẵn các thư viện lõi (bao gồm cả trình biên dịch C++ cho vLLM) trong hệ điều hành nền `python:3.10-slim`. Để đóng gói, Ban giám khảo chạy lệnh:
```bash
docker build -t team_name_innovator .