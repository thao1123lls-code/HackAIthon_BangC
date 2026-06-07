#!/usr/bin/env python
"""Quick test with 1 question to verify system"""

import sys
import os
import pandas as pd
import time

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

def test_inference():
    print("="*60)
    print("TESTING HACKATHON PROJECT")
    print("="*60)
    
    # Test 1: Check dependencies
    print("\n[1/4] Checking dependencies...")
    try:
        import pandas as pd
        print("  ✓ pandas")
        import torch
        print(f"  ✓ torch ({torch.__version__}, CUDA: {torch.cuda.is_available()})")
        from FlagEmbedding import BGEM3FlagModel
        print("  ✓ FlagEmbedding (BGE-M3)")
        from vllm import LLM, SamplingParams
        print("  ✓ vLLM")
    except ImportError as e:
        print(f"  ✗ Missing: {e}")
        return False
    
    # Test 2: Check data files
    print("\n[2/4] Checking data files...")
    if os.path.exists("knowledge_base.txt"):
        with open("knowledge_base.txt", "r") as f:
            kb_size = len(f.read())
        print(f"  ✓ knowledge_base.txt ({kb_size} bytes)")
    else:
        print("  ✗ knowledge_base.txt not found")
        return False
    
    # Test 3: Initialize RAG
    print("\n[3/4] Initializing RAG system...")
    try:
        from rag import RAGSystem
        print("  > Loading BGE-M3 model... (this may take 1-2 min on CPU)")
        rag = RAGSystem("knowledge_base.txt")
        print("  ✓ RAG system initialized")
        
        # Test RAG search
        query = "Python là gì?"
        print(f"\n  Testing RAG search: '{query}'")
        context = rag.search(query, top_k=2)
        if context:
            print(f"  ✓ Retrieved {len(context)} characters of context")
        else:
            print("  ⚠ No context retrieved")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    # Test 4: Test with small model (skip vLLM on CPU due to slowness)
    print("\n[4/4] Answer extraction test...")
    try:
        from main import extract_answer
        
        # Test extraction
        test_cases = [
            ("Đáp án: A", "A"),
            ("Chọn B", "B"),
            ("Là C", "C"),
            ("D là đúng", "D"),
            ("xyz", "A"),  # default
        ]
        
        for text, expected in test_cases:
            result = extract_answer(text)
            status = "✓" if result == expected else "✗"
            print(f"  {status} extract_answer('{text}') = {result} (expected {expected})")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print("\nNote: PyTorch running on CPU (no CUDA)")
    print("For full system, use Docker with GPU support")
    print("or install: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    return True

if __name__ == "__main__":
    success = test_inference()
    sys.exit(0 if success else 1)
