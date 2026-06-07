#!/usr/bin/env python
"""Simple test without GPU/models - just test core logic"""

import sys
import os
import pandas as pd
import re

print("="*60)
print("QUICK TEST (NO GPU/MODELS)")
print("="*60)

# Test 1: Check basic dependencies
print("\n[1/3] Checking basic dependencies...")
try:
    import pandas as pd
    print("  ✓ pandas")
    import re
    print("  ✓ regex")
except ImportError as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test 2: Check data files
print("\n[2/3] Checking data files...")
errors = []

if not os.path.exists("knowledge_base.txt"):
    errors.append("knowledge_base.txt not found")
else:
    with open("knowledge_base.txt", "r") as f:
        kb_content = f.read()
    kb_lines = len(kb_content.split('\n'))
    print(f"  ✓ knowledge_base.txt ({len(kb_content)} bytes, {kb_lines} lines)")

if not os.path.exists("data/public_test.csv"):
    errors.append("data/public_test.csv not found")
else:
    df = pd.read_csv("data/public_test.csv")
    print(f"  ✓ data/public_test.csv ({len(df)} rows, {len(df.columns)} columns)")
    print(f"    Columns: {list(df.columns)}")
    print(f"    First row: qid={df.iloc[0]['qid']}, question='{df.iloc[0]['question'][:50]}...'")

if errors:
    for e in errors:
        print(f"  ✗ {e}")

# Test 3: Test answer extraction function
print("\n[3/3] Testing answer extraction logic...")

def extract_answer(generated_text):
    """Test version of extract_answer"""
    # Layer 1: Primary pattern
    match = re.search(r'Đáp án:\s*([ABCD])', generated_text, re.IGNORECASE)
    if match: 
        return match.group(1).upper()
    
    # Layer 2: Alternative keywords
    match_2 = re.search(r'(chọn|là)\s*([ABCD])', generated_text, re.IGNORECASE)
    if match_2: 
        return match_2.group(2).upper()
    
    # Layer 3: Last ABCD character
    matches = re.findall(r'\b([ABCD])\b', generated_text)
    if matches: 
        return matches[-1].upper()
    
    # Layer 4: Default
    return "A"

test_cases = [
    ("Phân tích: câu hỏi là... Đáp án: A", "A", "Layer 1 (primary)"),
    ("Tôi chọn B vì...", "B", "Layer 2 (keyword 'chọn')"),
    ("Nó là C theo tôi", "C", "Layer 2 (keyword 'là')"),
    ("A sai, B sai, C sai, D đúng", "D", "Layer 3 (last ABCD)"),
    ("Không có đáp án rõ", "A", "Layer 4 (default)"),
]

all_passed = True
for text, expected, description in test_cases:
    result = extract_answer(text)
    passed = result == expected
    status = "✓" if passed else "✗"
    print(f"  {status} {description}")
    print(f"      Input: '{text[:50]}...'")
    print(f"      Expected: {expected}, Got: {result}")
    if not passed:
        all_passed = False

# Test 4: Output directory check
print("\n[BONUS] Checking output directory...")
if not os.path.exists("output"):
    os.makedirs("output")
    print("  ✓ Created output/ directory")
else:
    print("  ✓ output/ directory exists")

# Test 5: Sample output format
print("\n[BONUS] Testing output format...")
sample_output = pd.DataFrame({
    'qid': [1, 2, 3],
    'answer': ['A', 'B', 'C']
})
sample_output.to_csv("output/sample_pred.csv", index=False)
loaded = pd.read_csv("output/sample_pred.csv")
print(f"  ✓ Created sample output CSV")
print(f"    Rows: {len(loaded)}, Columns: {list(loaded.columns)}")
print(f"    Content:\n{loaded.to_string(index=False)}")

# Summary
print("\n" + "="*60)
if all_passed and not errors:
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print("\nSystem is ready to use!")
    print("\nNext steps:")
    print("  1. For GPU inference: docker run --gpus all [image]")
    print("  2. For local inference: python main.py (requires GPU)")
    print("  3. See QUICK_START.md for detailed instructions")
    sys.exit(0)
else:
    print("✗ SOME TESTS FAILED")
    print("="*60)
    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  - {e}")
    sys.exit(1)
