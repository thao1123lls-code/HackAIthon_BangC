import numpy as np
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from sklearn.metrics.pairwise import cosine_similarity

class RAGSystem:
    def __init__(self, kb_path="knowledge_base.txt"):
        print("Đang tải mô hình BGE-m3 (Chặng 1 - Truy xuất thô)...")
        # Khởi tạo BGE-m3 để quét vector nhanh
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        
        print("Đang tải mô hình Qwen-Rerank (Chặng 2 - Lọc tinh)...")
        # Khởi tạo Qwen-Rerank để làm giám khảo chấm điểm lại tài liệu
        self.reranker = FlagReranker('Qwen/Qwen-Rerank', use_fp16=True)
        
        # --- CẬP NHẬT MỚI: Kỹ thuật Chunking (Cắt nhỏ văn bản có gối đầu) ---
        def chunk_text(text, chunk_size=200, overlap=50):
            words = text.split()
            # Cắt thành các khối 200 từ, mỗi khối gối lên nhau 50 từ để giữ liền mạch ngữ cảnh
            return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), max(1, chunk_size - overlap))]

        # Đọc dữ liệu từ file kiến thức và áp dụng hàm cắt nhỏ
        with open(kb_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
            self.documents = chunk_text(raw_text, chunk_size=200, overlap=50)
            
        print(f"Đã cắt thành {len(self.documents)} đoạn tài liệu. Đang tạo Vector...")
        embeddings = self.model.encode(self.documents, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        self.doc_vectors = embeddings['dense_vecs']
        print("Hệ thống Advanced RAG Sẵn sàng!")

    def search(self, query, top_k=2):
        # Kiểm tra nếu file knowledge_base trống
        if not self.documents:
            return ""

        # ==========================================
        # CHẶNG 1: LỌC THÔ VỚI BGE-M3 (Lấy Top 10)
        # ==========================================
        query_embedding = self.model.encode([query], return_dense=True, return_sparse=False, return_colbert_vecs=False)['dense_vecs']
        similarities = cosine_similarity(query_embedding, self.doc_vectors)[0]
        
        # Lấy tối đa 10 tài liệu liên quan nhất
        retrieve_k = min(10, len(self.documents))
        top_indices = np.argsort(similarities)[-retrieve_k:][::-1]
        raw_docs = [self.documents[i] for i in top_indices]

        # ==========================================
        # CHẶNG 2: LỌC TINH VỚI QWEN-RERANK (Chốt Top K)
        # ==========================================
        # Tạo danh sách các cặp [Câu hỏi, Tài liệu] để gửi cho Reranker
        pairs = [[query, doc] for doc in raw_docs]
        
        # AI chấm điểm độ logic giữa câu hỏi và từng đoạn văn
        scores = self.reranker.compute_score(pairs)
        
        # Xử lý an toàn nếu thư viện chỉ trả về 1 số thay vì list (khi có 1 tài liệu)
        if isinstance(scores, float):
            scores = [scores]
            
        # Ghép điểm số với đoạn văn, sắp xếp từ cao xuống thấp
        scored_docs = list(zip(scores, raw_docs))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Trích xuất top_k đoạn văn xịn nhất (thường là 2) để mớm cho Qwen3.5
        final_k = min(top_k, len(scored_docs))
        best_docs = [doc for score, doc in scored_docs[:final_k]]
        
        return "\n".join(best_docs)

# Đoạn code test nhanh (chỉ chạy khi gọi trực tiếp file này)
if __name__ == "__main__":
    rag = RAGSystem()
    print("\n--- KẾT QUẢ TRA CỨU ---")
    print(rag.search("Cơ sở dữ liệu quan hệ là gì?"))