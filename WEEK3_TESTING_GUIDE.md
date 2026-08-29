# Quick Testing Guide: Week 3 NeMo Guardrails

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
python app/server.py
# Should see: "Uvicorn running on http://0.0.0.0:8000"
```

---

## Test Using curl or Python

### **Test 1: Check Guardrail Endpoint** 
Empty context → Should BLOCK

```bash
curl -X POST http://localhost:8000/check-guardrail \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of India?",
    "retrieved_context": "",
    "similarity_scores": [],
    "retrieved_results_count": 0
  }'
```

**Expected Response:**
```json
{
  "is_allowed": false,
  "reason": "No relevant document content retrieved",
  "score": 0.0,
  "context_quality": "empty",
  "message": "I can only answer questions based on the provided document.",
  "status_summary": "BLOCKED ❌ | Quality: empty | Score: 0.00 | Reason: ..."
}
```

---

### **Test 2: Valid Context** 
With retrieved context → Should ALLOW

```bash
curl -X POST http://localhost:8000/check-guardrail \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the revenue?",
    "retrieved_context": "[1] Text Chunk: The company revenue increased by 25 percent in 2025 due to strong market growth.",
    "similarity_scores": [0.85],
    "retrieved_results_count": 1
  }'
```

**Expected Response:**
```json
{
  "is_allowed": true,
  "reason": "Sufficient document context available",
  "score": 0.85,
  "context_quality": "high",
  "message": null,
  "status_summary": "ALLOWED ✅ | Quality: high | Score: 0.85 | Reason: ..."
}
```

---

### **Test 3: Combined Answer Endpoint**
Uses existing database → Retrieval + Guardrails

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the company revenue?",
    "route": "text",
    "top_k": 3
  }'
```

**If results found:**
```json
{
  "query": "What is the company revenue?",
  "retrieval_results": [...],
  "retrieved_context": "...",
  "guardrail_allowed": true,
  "guardrail_message": null,
  "guardrail_score": 0.85,
  "guardrail_context_quality": "high",
  "status_summary": "ALLOWED ✅"
}
```

**If no results (empty database):**
```json
{
  "guardrail_allowed": false,
  "guardrail_message": "I can only answer questions based on the provided document.",
  "guardrail_score": 0.0,
  "guardrail_context_quality": "empty"
}
```

---

## Test Using Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Test 1: Block out-of-scope question
response = requests.post(f"{BASE_URL}/check-guardrail", json={
    "query": "Write a Python program for a calculator",
    "retrieved_context": "",
    "similarity_scores": [],
    "retrieved_results_count": 0
})

result = response.json()
print(f"Allowed: {result['is_allowed']}")  # False
print(f"Message: {result['message']}")     # Block message
print(f"Quality: {result['context_quality']}")  # "empty"

# Test 2: Allow in-scope question
response = requests.post(f"{BASE_URL}/check-guardrail", json={
    "query": "What was mentioned in the document?",
    "retrieved_context": "Document content here...",
    "similarity_scores": [0.75],
    "retrieved_results_count": 1
})

result = response.json()
print(f"Allowed: {result['is_allowed']}")  # True
print(f"Message: {result['message']}")     # None
print(f"Quality: {result['context_quality']}")  # "medium" or "high"

# Test 3: Full retrieval + guardrails
response = requests.post(f"{BASE_URL}/answer", json={
    "query": "What is in the document?",
    "route": "text",
    "top_k": 3
})

result = response.json()
print(f"Question allowed: {result['guardrail_allowed']}")
print(f"Status: {result['status_summary']}")
```

---

## Test Cases Verification Matrix

| Test Case | Input | Expected | Actual |
|-----------|-------|----------|--------|
| **T1: Out-of-scope (empty)** | `retrieved_context=""` | BLOCK ❌ | |
| **T2: Out-of-scope (low score)** | `similarity_scores=[0.1]` | BLOCK ❌ | |
| **T3: In-scope (valid context)** | `context="revenue...", score=0.8` | ALLOW ✅ | |
| **T4: Low quality** | `context="x", len<50` | BLOCK ❌ | |
| **T5: High quality** | `context=long_text, score=0.9` | ALLOW ✅ | |

---

## Dashboard Integration (Optional)

To show guardrails status in Streamlit dashboard:

```python
# In app/dashboard.py, after retrieval_node() call
import requests

# After getting retrieval results
retrieved_state = retrieval_node(initial_state, client=client)

# Check guardrails
guardrail_response = requests.post(
    "http://localhost:8000/check-guardrail",
    json={
        "query": user_query,
        "retrieved_context": retrieved_state["retrieved_context"],
        "similarity_scores": retrieved_state["similarity_scores"],
        "retrieved_results_count": len(retrieved_state.get("retrieval_results", []))
    }
).json()

# Display result
if not guardrail_response["is_allowed"]:
    st.error(f"⚠️ {guardrail_response['message']}")
    st.info(f"Reason: {guardrail_response['reason']}")
else:
    st.success(f"✅ Query passed guardrail check (Quality: {guardrail_response['context_quality']})")
```

---

## Debugging

### Check if guardrails module is imported correctly
```python
from app.guardrails import should_answer_question
print("✅ Guardrails module imported successfully")
```

### Test guardrails directly in Python
```python
from app.guardrails.guardrails import check_query_in_scope

result = check_query_in_scope(
    query="Test question",
    retrieved_context="Test context with sufficient length",
    similarity_scores=[0.8],
    retrieved_results_count=1
)

print(f"Allowed: {result.is_allowed}")
print(f"Score: {result.score}")
print(f"Quality: {result.context_quality}")
```

### Check server logs
```
# Should see guardrail check logs
INFO:app.guardrails.guardrails:Guardrail check: Query='What is...', Context Quality=high, Score=0.85
```

---

## Success Criteria

✅ `/check-guardrail` endpoint responds correctly  
✅ Empty context returns BLOCK with correct message  
✅ Valid context returns ALLOW  
✅ Similarity scores affect context quality  
✅ Logs show guardrail decisions  
✅ Existing `/retrieve` endpoint still works  
✅ No Week 1/2 functionality affected  

---

## Common Issues

**Issue**: "Module not found: nemo_guardrails"
```bash
# Fix: Install dependency
pip install nemo-guardrails>=0.7.0
```

**Issue**: Guardrails always blocking
```python
# Check: Is context being provided?
print(retrieved_state["retrieved_context"])
# If empty, retrieval found no results
```

**Issue**: Server not starting
```bash
# Check Python version (3.8+)
python --version

# Check port 8000 is available
netstat -an | grep 8000  # On Windows/Mac
```

---

## Next Steps

1. **Dashboard Integration**: Display guardrail status in Streamlit
2. **LLM Integration**: Call `/answer` endpoint before generating response
3. **Custom Messages**: Modify `OUT_OF_SCOPE_MESSAGE` if needed
4. **Monitoring**: Log all guardrail decisions for analytics
5. **Tuning**: Adjust thresholds based on your documents
