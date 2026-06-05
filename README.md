# Giải pháp Bảng C - HackAIthon 2026

## 1. Build Docker Image
`docker build -t team_name_innovator .`

## 2. Run Container để test dự đoán
`docker run -v $(pwd)/data:/data -v $(pwd)/output:/output team_name_innovator`
