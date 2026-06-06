# Transformer Architecture — Summary

## Why Transformers exist

Before Transformers (pre-2017), language models used RNNs (Recurrent Neural Networks) which processed text sequentially — one token at a time, left to right. By the time the model reached the end of a long sentence, it had largely forgotten the beginning. Long range dependencies were very hard to learn.

Transformers solved this with one core idea: **let every token look at every other token simultaneously.**

The breakthrough paper: "Attention is All You Need" — Google, 2017. Every modern LLM (GPT, Claude, Gemini, Llama) is a Transformer.

---

## The full sequence — step by step

```
Raw text
    ↓
1. Tokenize         → text split into token IDs
    ↓
2. Embed            → token IDs → vectors (embedding matrix lookup)
    ↓
3. Add position     → positional encoding added to each vector
    ↓
4. Attention        → every token looks at every other, updates its vector
    ↓
5. Feedforward      → each token's vector processed independently
    ↓
   (steps 4–5 repeat for every layer — 32 to 96 times in large models)
    ↓
6. Output layer     → final vector → softmax → probability distribution
    ↓
Next token sampled (controlled by temperature)
```

Steps 4 and 5 together = one Transformer layer.

---

## Step 1 — Tokenize

Raw text is split into token IDs. The model never sees characters or words — only integers.

```
"The cat sat" → [464, 3797, 3332]
```

---

## Step 2 — Embed

Each token ID is looked up in the embedding matrix — a table of learned vectors. Each token becomes a list of numbers encoding its meaning.

```
464  ("The") → [0.2, -0.4, 0.8, ...]
3797 ("cat") → [0.3, -0.3, 0.7, ...]
3332 ("sat") → [-0.5, 0.8, -0.1, ...]
```

The embedding matrix is just another set of weights — initialized randomly, learned during training alongside everything else.

---

## Step 3 — Positional encoding

Attention processes all tokens in parallel — so it has no sense of order by default. "cat sat" and "sat cat" would look identical.

A positional signal is added to each embedding, encoding where in the sequence each token sits:

```
token embedding + position signal = final input vector
```

Now the model knows both what each token means and where it appears.

---

## Step 4 — Attention

The core innovation. Every token computes attention scores against every other token and updates its own vector by collecting weighted information from all others.

For each token, the model asks three questions:

```
What am I looking for?       → Query (Q)
What do I contain?           → Key (K)
What do I pass forward?      → Value (V)
```

High attention score → that token contributes a lot to the update.
Low attention score → that token is mostly ignored.

### What gets updated

Before attention — vectors carry isolated meaning:
```
"sat" → the act of sitting (generic)
```

After attention — vectors carry contextual meaning:
```
"sat" → a cat performing the act of sitting on a mat (contextual)
```

The same word "bank" starts with the same vector in every sentence. After attention:
```
"...bank to deposit money"  →  bank vector shifts toward financial institution
"...bank to catch fish"     →  bank vector shifts toward river bank
```

Same token. Same initial vector. Completely different vector after attention. Context rewrites meaning dynamically.

### Multi-head attention

The model doesn't run attention once per layer — it runs it many times in parallel. These are called **heads**. Each head learns to attend to different types of relationships:

```
Head 1 → subject-verb relationship (sat attends to cat)
Head 2 → verb-location relationship (sat attends to mat)
Head 3 → coreference (it attends to cat)
Head 4 → syntactic role (on attends to sat and mat)
...
```

Nobody programs these roles — each head discovers what's useful during training.

After all heads finish, their outputs are concatenated and projected back to the original vector size:

```
Head 1 output: [...]
Head 2 output: [...]   →  concatenate → linear projection → one vector per token
Head N output: [...]
```

### Scale

| Model | Layers | Heads per layer | Total heads |
|---|---|---|---|
| GPT-2 small | 12 | 12 | 144 |
| GPT-3 | 96 | 96 | 9,216 |
| Llama 2 7B | 32 | 32 | 1,024 |
| Llama 2 70B | 80 | 64 | 5,120 |

---

## Step 5 — Feedforward network

**Attention = communication between tokens. Feedforward = computation within a token.**

After attention, each token has a raw blend of gathered information. The feedforward network processes and refines it. Applied independently to each token — no communication between tokens here.

```
input vector
    → expand to larger vector (e.g. 4096 → 16384 dimensions)
    → ReLU (filter: keep positives, zero out negatives)
    → compress back down (16384 → 4096 dimensions)
    → output vector
```

**ReLU** is a simple gate:
```
if number > 0 → keep it
if number < 0 → set to 0
```

This non-linearity is what lets the network learn complex patterns. Without it, all layers would collapse into one simple transformation.

**Why expand then compress?** More working space for computation — like doing math on paper before writing the final answer.

### What feedforward is believed to store

Research suggests feedforward layers act as a knowledge store. When the model knows "Paris is the capital of France" — that fact likely lives in the feedforward weights. Attention figures out that the question is about Paris and France. Feedforward retrieves the stored fact.

### Why both attention and feedforward are needed

```
Attention alone    → can gather information but limited computation (weighted average)
Feedforward alone  → can compute but has no information from other tokens
Together           → gather relevant context, then compute something useful with it
```

---

## Step 6 — Stacked layers

After one layer (attention + feedforward), the process repeats:

```
Early layers (1–8)    → syntax, grammar, basic patterns
Middle layers (8–24)  → semantics, coreference, entity relationships
Late layers (24–96+)  → abstract reasoning, world knowledge, complex tasks
```

This division of labor is not programmed — it emerges from training. Deeper models are smarter because more layers = more levels of abstraction.

---

## Step 7 — Output

The final token's vector goes through a linear layer and softmax, producing a probability distribution over the entire vocabulary:

```
cat   → 35%
floor → 28%
sofa  → 18%
roof  → 10%
...
```

Temperature controls how to sample from this distribution. Then the process repeats for the next token.

---

## Why Transformers are powerful

| Property | Benefit |
|---|---|
| Parallelism | All tokens processed simultaneously — much faster training than sequential RNNs |
| Long range dependencies | Every token can attend to any other token regardless of distance |
| Scalability | More layers + heads + dimensions = smarter model, scales well with compute |
| Dynamic meaning | Same word gets different contextual representation based on surrounding tokens |

---

## Key vocabulary

| Term | Meaning |
|---|---|
| Transformer | Neural network architecture based on attention — backbone of all modern LLMs |
| Attention | Mechanism that lets every token look at every other token and update its vector |
| Query / Key / Value | The three projections used in attention — what am I looking for, what do I have, what do I pass forward |
| Attention score | How much one token should attend to another — determines contribution to the update |
| Multi-head attention | Running attention multiple times in parallel, each head learning different relationships |
| Attention head | One parallel instance of attention — learns one type of relationship |
| Positional encoding | Signal added to embeddings so the model knows token order |
| Feedforward network | Small neural network applied per token after attention — adds computational power |
| ReLU | Simple activation function — keeps positive values, zeros out negatives |
| Layer | One block of attention + feedforward. Stacked many times in a Transformer |
| Context window | Maximum number of tokens the model can attend to at once (covered in topic 6) |
