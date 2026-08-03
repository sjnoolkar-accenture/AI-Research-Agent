# Building Your Own RAG (Retrieval-Augmented Generation) Project
## A Step-by-Step Guide for C# Developers Learning Python

---

## Table of Contents
1. [What is RAG?](#what-is-rag)
2. [Project Overview](#project-overview)
3. [Prerequisites](#prerequisites)
4. [Step-by-Step Implementation](#step-by-step-implementation)
5. [Python Basics for C# Developers](#python-basics-for-c-developers)
6. [Dataset Recommendations](#dataset-recommendations)
7. [Troubleshooting](#troubleshooting)

---

## What is RAG?

**RAG = Retrieval-Augmented Generation**

It's a system that:
1. **Retrieves** relevant information from a knowledge base (vector database)
2. **Augments** an LLM's prompt with that information
3. **Generates** an answer using both the retrieved context and the LLM

**Why RAG?**
- Reduces hallucinations (AI making up facts)
- Keeps answers current (uses your data, not just training data)
- Cost-effective (doesn't require retraining the model)

**Real-world analogy:**
- Without RAG: "What's in the company handbook?" → LLM guesses based on training data
- With RAG: LLM searches handbook first, then answers based on actual content

---

## Project Overview

Your project will have **3 parts**:

### Part 1: Data Preparation & Vector Database
- Load your dataset (JSON, CSV, or API)
- Clean and structure the data
- Convert text → embeddings (semantic vectors)
- Store in a vector database (Chroma)

### Part 2: RAG Agent with Memory & Tools
- Retrieve relevant documents from the vector DB
- Add persistent memory (agent learns over time)
- Support web search fallback
- Structured output (JSON validation with Pydantic)
- Custom analysis tools (domain-specific)

### Part 3: Visualization & Analysis
- Dashboard showing data composition
- Embedding space visualization (PCA)
- Retrieval trace (how well queries match data)

---

## Prerequisites

### Python Environment
```bash
# Create virtual environment (like .NET virtual env)
python -m venv project-env

# Activate (PowerShell)
.\project-env\Scripts\Activate.ps1

# Install required packages
pip install chromadb openai pydantic python-dotenv matplotlib scikit-learn requests
```

### API Keys
1. **OpenAI API Key** (for embeddings + LLM)
   - Sign up: https://platform.openai.com
   - Get key: https://platform.openai.com/account/api-keys
   - Cost: ~$0.02 per 1K embeddings

2. **Tavily API Key** (for web search, optional)
   - Sign up: https://tavily.com
   - Free tier available

3. Create `.env` file:
   ```
   OPENAI_API_KEY=sk-...your-key...
   TAVILY_API_KEY=your-tavily-key
   ```

---

## Step-by-Step Implementation

### **STEP 1: Choose Your Dataset**

Pick a domain you know well (C# → maybe programming topics, game APIs, tech docs):

**Option A: JSON File (Easiest)**
```json
[
  {
    "id": 1,
    "title": "Entity Framework Basics",
    "category": "ORM",
    "content": "Entity Framework is...",
    "tags": ["C#", "Database", "ORM"]
  },
  ...
]
```

**Option B: CSV File**
```csv
id,title,category,content,tags
1,Entity Framework Basics,ORM,"Entity Framework is...",C#;Database;ORM
```

**Option C: API or Web Scraping** (advanced)

**Recommended for beginners:** Start with 20-50 documents in JSON format

---

### **STEP 2: Create Part 1 - Data Preparation Notebook**

Create `Project_Part1_DataPrep.ipynb`

#### Cell 1: Load Dependencies
```python
import os
import json
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()  # Load API keys from .env
```

#### Cell 2: Load Your Dataset
```python
# Load JSON file
with open("your_dataset.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"Loaded {len(raw_data)} documents")
print(f"First item: {raw_data[0]}")  # Inspect structure
```

#### Cell 3: Prepare Embeddings Client
```python
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small",  # Small = cheap & fast
)

print("Embedding function ready")
```

#### Cell 4: Create Vector Database
```python
# Connect to persistent database (saves to disk)
chroma_client = chromadb.PersistentClient(path="knowledge_base_db")

# Create collection (like a database table)
kb_collection = chroma_client.get_or_create_collection(
    name="my_knowledge",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},  # Distance metric
)

print(f"Collection ready: {kb_collection.count()} documents")
```

#### Cell 5: Process & Store Documents
```python
def prepare_documents(raw_data):
    """Convert raw data into format for vector DB"""
    docs = []
    metas = []
    ids = []
    
    for i, item in enumerate(raw_data):
        # Combine text fields into a document
        document_text = f"""
Title: {item.get('title', '')}
Category: {item.get('category', '')}
Content: {item.get('content', '')}
Tags: {', '.join(item.get('tags', []))}
        """.strip()
        
        docs.append(document_text)
        
        # Store metadata separately
        metas.append({
            "title": item.get("title", ""),
            "category": item.get("category", ""),
            "tags": ", ".join(item.get("tags", [])),
        })
        
        ids.append(f"doc_{i:03d}")
    
    return docs, metas, ids

docs, metas, ids = prepare_documents(raw_data)

# Upload to vector database (auto-generates embeddings)
kb_collection.upsert(ids=ids, documents=docs, metadatas=metas)

print(f"✓ Stored {len(docs)} documents in vector database")
```

#### Cell 6: Test Retrieval
```python
# Test: retrieve documents similar to a query
test_query = "How do I use Entity Framework?"
results = kb_collection.query(query_texts=[test_query], n_results=3)

print(f"Query: {test_query}\n")
for i, (doc, meta, distance) in enumerate(zip(
    results["documents"][0], 
    results["metadatas"][0], 
    results["distances"][0]
)):
    print(f"{i+1}. {meta['title']}")
    print(f"   Similarity: {1 - distance:.2%}")
    print(f"   Preview: {doc[:100]}...\n")
```

---

### **STEP 3: Create Part 2 - RAG Agent Notebook**

Create `Project_Part2_Agent.ipynb`

#### Cell 1-2: Setup (same as Part 1)
```python
# Load dependencies, API keys, and connect to DB
# (copy from Part 1 cells 1-4)
```

#### Cell 3: Define RAG Agent Class
```python
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

client = OpenAI(api_key=OPENAI_API_KEY)

class Answer(BaseModel):
    """Structured response from the agent"""
    answer: str = Field(..., description="The actual answer")
    confidence: float = Field(..., ge=0, le=1, description="0-1 confidence")
    source_documents: List[str] = Field(default_factory=list, description="Document titles used")
    source_type: str = Field(..., description="'knowledge_base' or 'web_search'")

def ask_agent(query: str) -> Answer:
    """
    1. Retrieve relevant documents
    2. Send to LLM with context
    3. Parse structured response
    """
    
    # Step 1: Retrieve
    results = kb_collection.query(query_texts=[query], n_results=3)
    context = "\n".join(results["documents"][0])
    source_docs = [m["title"] for m in results["metadatas"][0]]
    
    # Step 2: Check quality (is retrieval good enough?)
    top_distance = results["distances"][0][0]
    confidence = max(0.0, 1.0 - top_distance)
    
    # Step 3: Call LLM
    system_prompt = """You are a helpful assistant. Answer using ONLY the provided context.
    Respond as JSON with keys: answer, confidence (0-1), source_documents (list), source_type."""
    
    user_prompt = f"Question: {query}\n\nContext:\n{context}"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Fast & cheap model
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    
    # Step 4: Parse response
    import json
    payload = json.loads(response.choices[0].message.content)
    payload["source_documents"] = source_docs
    payload["source_type"] = "knowledge_base"
    
    return Answer(**payload)

print("Agent initialized")
```

#### Cell 4: Test Agent
```python
result = ask_agent("What is Entity Framework?")
print("Answer:", result.answer)
print("Confidence:", result.confidence)
print("Sources:", result.source_documents)
```

---

### **STEP 4: Add Memory (Agent Learns)**

```python
from collections import Counter
import time

# Second vector database for memory
memory_collection = chroma_client.get_or_create_collection(
    name="agent_memory",
    embedding_function=embedding_fn,
)

def remember(query: str, fact: str):
    """Save learned fact"""
    mem_id = f"mem_{int(time.time() * 1000)}"
    memory_collection.upsert(
        ids=[mem_id],
        documents=[fact],
        metadatas=[{"query": query, "timestamp": time.time()}],
    )

def recall(query: str):
    """Check memory before searching"""
    if memory_collection.count() == 0:
        return []
    results = memory_collection.query(query_texts=[query], n_results=2)
    return results["documents"][0] if results["distances"][0][0] < 0.5 else []

# Use in agent:
def ask_agent_with_memory(query: str) -> Answer:
    # Check memory first
    memory_facts = recall(query)
    if memory_facts:
        # Use memory instead of search
        context = "\n".join(memory_facts)
        source_type = "memory"
    else:
        # Search knowledge base
        results = kb_collection.query(query_texts=[query], n_results=3)
        context = "\n".join(results["documents"][0])
        source_type = "knowledge_base"
        # Remember this for next time
        remember(query, context)
    
    # Rest of agent logic...
    # (same as before)
```

---

### **STEP 5: Add Custom Tools**

Example: sentiment analysis of your documents

```python
def analyze_sentiment(text: str) -> dict:
    """Simple sentiment analysis"""
    positive = {"great", "excellent", "good", "amazing", "helpful"}
    negative = {"bad", "poor", "awful", "useless", "confusing"}
    
    words = text.lower().split()
    pos_count = sum(1 for w in words if w in positive)
    neg_count = sum(1 for w in words if w in negative)
    
    score = (pos_count - neg_count) / max(1, pos_count + neg_count)
    
    return {
        "sentiment": "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral",
        "score": score,
    }

# Analyze all documents
for doc in docs:
    sentiment = analyze_sentiment(doc)
    print(f"{sentiment['sentiment']}: {doc[:50]}...")
```

---

### **STEP 6: Visualization**

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Get embeddings from DB
kb_data = kb_collection.get(include=["embeddings", "metadatas"])
embeddings = np.array(kb_data["embeddings"])

# Reduce to 2D for visualization
coords = PCA(n_components=2).fit_transform(embeddings)

# Plot
plt.figure(figsize=(12, 8))
categories = [m["category"] for m in kb_data["metadatas"]]
unique_cats = set(categories)

for cat in unique_cats:
    idx = [i for i, c in enumerate(categories) if c == cat]
    plt.scatter(coords[idx, 0], coords[idx, 1], label=cat, s=100)

plt.legend()
plt.title("Document Embedding Space")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.savefig("embedding_space.png", dpi=150)
plt.show()
```

---

## Python Basics for C# Developers

### Quick Reference

| C# | Python | Notes |
|---|---|---|
| `List<T>` | `list` | Python lists are dynamic |
| `Dictionary<K,V>` | `dict` | Similar syntax |
| `class MyClass { }` | `class MyClass:` | Indentation matters |
| `.` | `.` | Same |
| `async/await` | `async/await` | Same concept |
| `try/catch` | `try/except` | Similar |
| `interface` | No direct equivalent | Use `abc` module |
| `null` | `None` | Python's null |
| `string.Format()` | f-strings: `f"{var}"` | f-strings are cleaner |
| `using` | `with` | Context managers |

### Key Differences

**No Type Declarations (usually)**
```csharp
// C#
string name = "John";
int age = 30;
```

```python
# Python - types are inferred
name = "John"
age = 30
```

**Indentation = Code Blocks**
```python
# Python uses indentation (no braces)
if age > 18:
    print("Adult")
    print("Can vote")
else:
    print("Minor")
```

**List Comprehensions** (powerful in Python)
```python
# Create list of squares
squares = [x**2 for x in range(10)]  # [0, 1, 4, 9, ...]

# Filter even numbers
evens = [x for x in range(10) if x % 2 == 0]  # [0, 2, 4, 6, 8]
```

**Dictionary Access**
```python
person = {"name": "John", "age": 30}
print(person["name"])  # "John"
print(person.get("email", "unknown"))  # "unknown" (default)
```

---

## Dataset Recommendations

### For C# Developers:
1. **C# Documentation Topics** (20-30 docs)
   - Entity Framework
   - LINQ
   - Async/Await
   - Dependency Injection
   - etc.

2. **Project Management Tips** (30-50 docs)
   - Agile practices
   - Code reviews
   - Testing strategies
   - etc.

3. **Internal Company Knowledge** (most valuable!)
   - Code standards
   - Architecture patterns
   - Business rules
   - Past project learnings

### Dataset Format (JSON)
```json
[
  {
    "id": 1,
    "title": "LINQ Query Syntax",
    "category": "LINQ",
    "content": "LINQ provides a query syntax similar to SQL...",
    "difficulty": "intermediate",
    "tags": ["C#", "LINQ", "Queries"]
  },
  {
    "id": 2,
    "title": "Async Patterns",
    "category": "Async",
    "content": "Use async/await for non-blocking operations...",
    "difficulty": "advanced",
    "tags": ["C#", "Async", "Concurrency"]
  }
]
```

---

## Troubleshooting

### Issue: "chromadb.errors.InvalidDimensionException"
**Cause:** Embedding model mismatch
**Fix:** Ensure you use same embedding model throughout:
```python
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    model_name="text-embedding-3-small",  # Must match
)
```

### Issue: "OpenAIError: Missing API key"
**Cause:** API key not in `.env` or not loaded
**Fix:**
```python
from dotenv import load_dotenv
load_dotenv()  # Must call this first
```

### Issue: Slow retrieval
**Cause:** Too many documents or large embeddings
**Fix:** Start with <100 documents, use `text-embedding-3-small` model

### Issue: Poor answer quality
**Cause:** Bad retrieval or weak context
**Fix:**
1. Check retrieval similarity scores (should be < 0.4)
2. Improve document quality/structure
3. Use longer, more detailed documents

---

## Next Steps

1. **Choose your dataset** → Collect 20-30 documents
2. **Run Part 1** → Build vector database
3. **Run Part 2** → Create agent and test
4. **Customize** → Add your own tools/analysis
5. **Document** → Explain what you learned and created
6. **Submit** → Include disclosure of AI assistance used

---

## Resources

- **Chroma Docs:** https://docs.trychroma.com
- **OpenAI API:** https://platform.openai.com/docs
- **Pydantic:** https://docs.pydantic.dev
- **Python for C# Developers:** https://python-guide.readthedocs.io

---

**Good luck building your RAG project!** 🚀
