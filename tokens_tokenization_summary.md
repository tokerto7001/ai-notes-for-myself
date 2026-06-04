# Tokens & Tokenization — Summary

## What is a token?

A token is not a word. It is a **chunk of text** — could be a whole word, part of a word, or a single character. Models never see raw text — they only ever see token IDs (integers).

```
"cat"          → ["cat"]                   (1 token)
"unbelievable" → ["un", "believ", "able"]  (3 tokens)
"ChatGPT"      → ["Chat", "G", "PT"]       (3 tokens)
" hello"       → [" hello"]                (space is part of the token)
"1234567"      → ["123", "456", "7"]       (3 tokens)
```

After tokenization, every token maps to an integer ID:

```
"the" → 464
"cat" → 3797
"sat" → 3332
```

The neural network only ever sees sequences of integers — never raw text.

---

## Why not just use words?

Three problems with word-based vocabularies:

- **Vocabulary explosion** — hundreds of thousands of words plus names, slang, technical terms, other languages, code
- **Unknown words** — any word not in the vocabulary breaks the model
- **Inefficiency** — "run", "running", "runner" treated as completely unrelated

Tokens solve all three. A fixed vocabulary of ~50,000 tokens can represent virtually any text by breaking unknown words into known subword pieces.

---

## BPE — Byte Pair Encoding

Most modern LLMs use BPE. Two phases:

### Phase 1 — Training the tokenizer (done once, before LLM training)

1. Start with individual characters
2. Count all character pairs across the entire corpus
3. Merge the most frequent pair into a single token
4. Repeat until target vocabulary size is reached (typically 50k–100k tokens)

```
"lowest" → ["l","o","w","e","s","t"]   start
         → ["l","o","w","es","t"]      merge e+s → es (most frequent)
         → ["l","o","w","est"]         merge es+t → est
         → ["low","est"]               merge l+o+w → low
         → ["lowest"]                  merge low+est → lowest (if frequent enough)
```

Frequent words end up as single tokens. Rare words stay as subword pieces. Individual characters are always kept as a fallback.

### Phase 2 — Using the tokenizer (at inference and during LLM training)

Apply the learned merge table greedily to new text — same merges, same order, every time. Deterministic.

---

## The full pipeline before an LLM can be trained

```
1. Collect raw text data (internet, books, code...)
        ↓
2. Train the tokenizer on that data (BPE)
        ↓
3. Tokenize the entire dataset using the trained tokenizer
        ↓
4. Train the LLM on the tokenized data
```

The tokenizer must be ready before LLM training starts. The LLM never sees raw text.

---

## Cost comparison

| Step | Cost |
|---|---|
| Data collection | Medium |
| Tokenizer training | Low (CPU, hours) |
| Dataset tokenization | Medium |
| LLM pretraining | Enormous (thousands of GPUs, months) |
| Fine-tuning / RLHF | High |
| Deployment | Ongoing |

---

## Each model family has its own tokenizer

Different companies train different tokenizers on different data with different vocabulary sizes. Same algorithm (BPE), different trained output.

```
"unbelievable"
GPT-4:  ["un", "believ", "able"]    → 3 tokens
Llama:  ["un", "believe", "able"]   → 3 tokens (different split)
Claude: ["unbelievable"]            → 1 token (if in vocabulary)
```

This is why token counts differ between providers — and why context window sizes aren't directly comparable across models.

---

## Tokenizer is fixed forever after LLM training

The model's weights are built around specific token IDs. Changing the tokenizer after training would break billions of weight values. You'd have to retrain the entire LLM from scratch.

This is why a new tokenizer = a new model family, not a minor update.

---

## Why this matters practically

**Cost** — APIs charge per token. Know your token counts.

**Context window** — measured in tokens, not words. Rough rule of thumb:
```
1 token ≈ ¾ of an English word ≈ 4 characters

100 words  ≈ 130 tokens
1000 words ≈ 1300 tokens
A novel    ≈ 100,000+ tokens
```

**Quirky model behavior** — explained by tokenization:
- Struggling to count letters ("how many r's in strawberry") — the model sees one token, not individual letters
- Weird behavior with rare words — split into subword pieces with weak associations
- Better at English than other languages — English dominates training data so English words get merged aggressively into fewer tokens

**Tokenization is case and space sensitive:**
```
"cat"  → one token
" cat" → different token (leading space included)
"Cat"  → possibly different token
```

Slight behavior differences from capitalization or spacing are often explained by this.

---

## Key vocabulary

| Term | Meaning |
|---|---|
| Token | A chunk of text — word, subword, or character — that the model processes |
| Token ID | The integer assigned to a token in the vocabulary |
| Vocabulary | The fixed set of all tokens a model knows (typically 50k–100k) |
| BPE | Byte Pair Encoding — the algorithm used to train most modern tokenizers |
| Merge table | The ordered list of character merges BPE learned during tokenizer training |
| Subword | A piece of a word that becomes its own token when the full word is rare |
| Context window | The maximum number of tokens a model can process at once |
