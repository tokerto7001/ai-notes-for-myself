# Context Windows & Memory — Summary

## What is the context window?

The context window is the maximum number of tokens the model can see at once. Everything inside it the model can attend to. Everything outside it the model is completely blind to.

```
Context window = the model's entire working memory for one conversation
```

There is no other memory. No background storage. No persistent awareness. Just the tokens currently in the window.

---

## What lives inside the context window

```
System prompt
    +
Full conversation history (all previous messages)
    +
Current message
    +
Model's response (as it generates)
```

All of that counts toward the limit. Together.

---

## Current context window sizes

```
GPT-4o          → 128,000 tokens  (~96,000 words)
Claude 3.5      → 200,000 tokens  (~150,000 words)
Gemini 1.5 Pro  → 1,000,000 tokens
```

---

## What happens when you hit the limit

The model doesn't crash. It silently drops the oldest content:

```
[message 1]  ← falls out, model has no awareness of this
[message 2]  ← falls out
[message 3]  ← still visible
[message 4]  ← still visible
[current]    ← still visible
```

The model has no awareness that it forgot anything. It just doesn't have access to that information anymore. This is why long conversations degrade — the model loses early context and becomes inconsistent.

---

## The model has no memory between conversations

When a conversation ends — the context window is wiped. Completely. Start a new conversation and the model knows nothing about the previous one. There is no persistent memory unless you explicitly build it.

---

## Memory is an engineering problem, not a model problem

Engineers solve this in several ways:

**Stuff the context** — include everything relevant in the system prompt. Simple but hits limits fast.

**Summarization** — when conversation gets long, summarize older parts and replace them with the summary. Lossy but keeps the window manageable.

**RAG** — store information in a vector database. Retrieve relevant chunks at query time and inject into the context window.

**External memory stores** — save structured facts to a database. Inject relevant facts into the system prompt each time.

All of these are workarounds for the same fundamental constraint — the model only knows what's currently in its window.

---

## Longer context doesn't mean better performance

Models perform best on information at the beginning and end of the context. Information in the middle gets relatively less attention. This is called the **lost in the middle** problem:

```
Start of context   → strong attention
Middle of context  → weaker attention  ← danger zone
End of context     → strong attention
```

A model may miss information buried in the middle even if it technically fits in the window.

---

## The quadratic scaling problem

Attention computes scores between every token and every other token:

```
1,000 tokens    →  1,000 × 1,000     =  1,000,000 attention pairs
10,000 tokens   →  10,000 × 10,000   =  100,000,000 attention pairs
100,000 tokens  →  100,000 × 100,000 =  10,000,000,000 attention pairs
```

Double the context → four times the computation. This is called **quadratic scaling** and is a fundamental challenge with the Transformer architecture. It is why:

- Context windows were small (2,048 tokens) for years
- Long context requests are slow and expensive
- A 100k token request is ~100x more expensive than a 10k token request in raw attention computation

---

## How modern models handle quadratic scaling

**Flash Attention** — reimplements attention in memory-efficient chunks. Same math, much less memory. Used by almost every modern model.

**Sparse attention** — each token only attends to a subset of other tokens instead of all of them. Speed gains at some accuracy cost.

**Grouped Query Attention (GQA)** — multiple attention heads share Key and Value matrices. Reduces KV cache size significantly. Used in Llama 2 and most modern open source models.

---

## KV cache — why long context is expensive

For each layer, the model computes Key and Value matrices for every token. These are stored in GPU memory for the entire request:

```
200,000 tokens × 96 layers × Key and Value matrices = enormous GPU memory
```

Longer context = more GPU memory = higher cost. This is why context window size and inference cost are directly linked.

---

## What this means for your work as an AI engineer

- Design prompts to be token-efficient — every token costs money and window space
- Put the most important information at the start or end of context, not buried in the middle
- Don't rely on the model remembering things across sessions — build explicit memory if needed
- Chunk documents thoughtfully in RAG — retrieve only what's relevant, don't dump everything in
- Monitor token usage in production — costs scale with context size

---

## Key vocabulary

| Term | Meaning |
|---|---|
| Context window | Maximum tokens the model can process at once — its entire working memory |
| KV cache | Key and Value matrices stored in GPU memory during inference — grows with context length |
| Lost in the middle | Phenomenon where models attend less to information buried in the middle of long contexts |
| Quadratic scaling | Attention computation grows as the square of context length — doubling context = 4x compute |
| Flash Attention | Memory-efficient attention algorithm — same results, dramatically less GPU memory |
| Sparse attention | Attention variant where tokens only attend to a subset of other tokens |
| Grouped Query Attention | Multiple attention heads sharing Key/Value matrices to reduce memory usage |
