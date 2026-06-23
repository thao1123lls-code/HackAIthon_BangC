# Sơ đồ kiến trúc (dùng để đưa vào GitHub)

## Mermaid flowchart (copy/paste vào README)

> GitHub sẽ render Mermaid nếu repo bật Mermaid.

```mermaid
flowchart TD
    A[INPUT: File JSON chứa câu hỏi] --> B[HỆ THỐNG ADVANCED RAG]

    subgraph RAG[RAG System - 2 Stages]
        B1[Chặng 1: Vector Retrieval (BGE-M3)
- Lấy Top 10] --> B2[Chặng 2: Semantic Reranking (Qwen-Rerank)
- Chọn Top 2]
    end

    B --> C[PROMPT + CONTEXT + QUESTION
(Reflective CoT Format)]
    C --> D[LLM Engine (vLLM)
- Temperature = 0.0 (Greedy)
- Max sequence = 2048]
    D --> E[ANSWER EXTRACTION (3-layer fallback)
- Layer 1: "Đáp án: X"
- Layer 2: "Chọn/Là X"
- Layer 3: ký tự ABCD cuối cùng]
    E --> F[OUTPUT: Ghi vào file submission.csv]
```

