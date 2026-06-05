# Embeddings — Summary

## What is an embedding?

An embedding is a **list of numbers (a vector)** that represents the meaning of a token, word, sentence, or document. Models need more than just a token ID (an arbitrary integer) — they need a representation that captures meaning.

```
"cat"  → [0.2, -0.4, 0.8, 0.1, -0.7, ...]  (hundreds of numbers)
"dog"  → [0.3, -0.3, 0.7, 0.2, -0.6, ...]  (similar — both animals)
"car"  → [-0.5, 0.8, -0.1, 0.9, 0.3, ...]  (very different)
```

---

## Meaning becomes geometry

Similar meanings → similar vectors → close together in space. This is the core insight.

```
"cat" and "dog"    → vectors are close      (both animals, both pets)
"cat" and "car"    → vectors are far apart   (nothing in common)
"king" and "queen" → vectors are close       (both royalty)
```

Because meaning is geometry, you can do math on meanings:

```
king - man + woman = queen
paris - france + italy = rome
```

Nobody designed this. It emerged from training.

---

## The full journey inside the network

Embeddings are just the entry point — a lot more happens before probabilities are produced:

```
Tokens → Embeddings → [Transformer layers] → Probabilities
```

| Stage | What happens |
|---|---|
| Embeddings | Token IDs converted to vectors. Tokens don't know about each other yet. |
| Transformer layers | Every token looks at every other token and updates its vector (attention). Rich contextual meaning is built up. |
| Probabilities | Final vector converted to probability distribution over vocabulary via softmax. |

Embeddings alone don't decide the next token. They are the starting point. The Transformer layers do the heavy work.

---

## Where do embeddings come from?

They are learned during training — not designed by hand.

The network contains an **embedding matrix** — a lookup table mapping every token ID to a vector. It is just another set of weights, initialized randomly like everything else. Backpropagation updates it alongside all other weights.

```
Early training:
"cat" → [0.91, -0.23, 0.44, ...]  ← random, meaningless
"dog" → [0.12, 0.78, -0.55, ...]  ← random, no relation to cat

Late in training:
"cat" → [0.2, -0.4, 0.8, ...]     ← learned, meaningful
"dog" → [0.3, -0.3, 0.7, ...]     ← now close to cat
```

"cat" and "dog" became close because they appeared in similar contexts billions of times. The model was never told they are similar — it figured it out because representing them similarly reduces loss.

---

## Semantic closeness is a side effect, not the goal

The training goal is next token prediction. Semantic structure in embedding space is a byproduct:

```
Goal:        predict next token correctly
Side effect: geometry of meaning emerges in embedding space
```

Nobody defines what "meaning" is. Nobody labels semantic relationships. The model discovers the structure of meaning on its own — because that structure is what's most useful for predicting text.

---

## Handling unknown words

If a brand new word appears — say "Zyphronix" — the tokenizer breaks it into known subword pieces:

```
"Zyphronix" → ["Z", "yph", "ron", "ix"]
```

Each subword already has an embedding from training. The model combines them. It won't understand the specific word, but it can process it without breaking. This is a core reason subword tokenization exists.

---

## Two types of embeddings

**Token embeddings** — vector for a single token inside the model during processing. Part of the model's internal machinery. You never interact with these directly.

**Sentence / document embeddings** — a single vector representing an entire sentence or paragraph. Produced by a separate embedding model (e.g. OpenAI's `text-embedding-ada-002`). This is what you use in RAG and Qdrant.

Both are built on the same principle — meaning as geometry — but they serve different purposes.

---

## Dimensions

Embedding vectors have a fixed number of dimensions determined by the model:

```
OpenAI text-embedding-ada-002   → 1536 dimensions
OpenAI text-embedding-3-small   → 1536 dimensions
OpenAI text-embedding-3-large   → 3072 dimensions
```

Each dimension loosely corresponds to some learned feature of meaning — but you can't interpret individual dimensions. The geometry emerges from training, not from design.

---

## Parameters = weights = the same thing

Every adjustable number in the network is a parameter (also called a weight). They live across:

```
Embedding matrix        → token × dimension weights
Transformer layers      → attention + feedforward weights (per layer)
Final output layer      → vocabulary projection weights
```

7 billion parameters = 7 billion individual numbers, all tuned through training.

More parameters = more capacity to encode complex patterns:

```
1B parameters   → simpler patterns, shallower knowledge
7B parameters   → noticeably smarter, better reasoning
70B parameters  → much stronger, complex tasks
700B+           → frontier model territory
```

Parameters also determine model size on disk:

```
7B  × 2 bytes (16-bit) = ~14GB minimum
70B × 2 bytes          = ~140GB minimum
```

This is why running large models locally requires serious hardware. Quantization (compressing to 4-bit or 8-bit) reduces memory at some cost to quality.

---

## Why embeddings matter for your work

Everything in RAG relies on embeddings:

- **RAG** — embed documents, embed query, find closest documents by vector similarity
- **Semantic search** — find results by meaning not keywords
- **Clustering** — group similar documents together
- **Recommendations** — find items similar to what a user liked

Two completely different sentences with zero shared words can be recognized as meaning the same thing — because their vectors are close:

```
"How do I reset my password?"
"I forgot my login credentials"
```

Different words. Same meaning. Close vectors. Keyword search misses this. Embedding search catches it.

---

## Key vocabulary

| Term | Meaning |
|---|---|
| Embedding | A vector (list of numbers) representing the meaning of a token or text |
| Embedding matrix | Lookup table inside the model mapping token IDs to vectors — learned during training |
| Vector | A list of numbers representing a point in high-dimensional space |
| Dimensions | The length of the embedding vector — determines the richness of representation |
| Semantic similarity | Closeness of meaning — reflected as closeness of vectors in embedding space |
| Cosine similarity | Common metric for measuring how close two vectors are (used in Qdrant) |
| Parameter / weight | One adjustable number in the neural network — same thing, two names |
| Quantization | Compressing model weights to lower precision (4-bit, 8-bit) to reduce memory usage |
