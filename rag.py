import numpy as np
from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics.pairwise import cosine_similarity

class RAGSystem:
    def __init__(self, kb_path="knowledge_base.txt"):
        print("Đang tải mô hình BGE-m3...")
        # Khởi tạo mô hình BGE-m3 theo chuẩn quy định
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        
        # Đọc dữ liệu từ file kiến thức
        with open(kb_path, 'r', encoding='utf-8') as f:
            self.documents = [line.strip() for line in f if line.strip()]
            
        print(f"Đã nạp {len(self.documents)} tài liệu. Đang tạo Vector...")
        # Biến tài liệu thành vector. BGE-m3 trả về dict, ta lấy phần 'dense_vecs'
        embeddings = self.model.encode(self.documents, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        self.doc_vectors = embeddings['dense_vecs']
        print("Sẵn sàng tra cứu!")

    def search(self, query, top_k=1):
        # Biến câu hỏi thành vector
        query_embedding = self.model.encode([query], return_dense=True, return_sparse=False, return_colbert_vecs=False)['dense_vecs']
        
        # So sánh độ tương đồng giữa câu hỏi và toàn bộ tài liệu
        similarities = cosine_similarity(query_embedding, self.doc_vectors)[0]
        
        # Lấy ra index của top_k tài liệu liên quan nhất
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Trả về các đoạn văn bản tương ứng
        results = [self.documents[i] for i in top_indices]
        return " ".join(results)

# Đoạn code test nhanh (chỉ chạy khi gọi trực tiếp file này)
if __name__ == "__main__":
    rag = RAGSystem()
    print(rag.search("Cơ sở dữ liệu quan hệ là gì?"))
    