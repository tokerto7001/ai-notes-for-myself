# Inference Time Concepts — Summary

## What happens at inference time

The model has finished training. Weights are frozen. You send a prompt. The model generates one token at a time:

```
1. Run full forward pass on all current tokens
2. Get probability distribution over vocabulary
3. Sample one token from that distribution
4. Append it to the sequence
5. Repeat until done
```

How you sample in step 3 is what inference time concepts are all about.

---

## Temperature

Controls how strictly the model follows its probability distribution. Divides the raw scores before softmax — low temperature sharpens the distribution, high temperature flattens it.

```
temperature: 0    → deterministic — same input always same output
temperature: 0.3  → conservative — code, structured data, factual Q&A
temperature: 0.7  → balanced — most conversational use cases
temperature: 1.0  → creative — brainstorming, creative writing
```

---

## Top-K sampling

Only sample from the top K most probable tokens. Everything else gets zero probability.

```
K = 5:
mat    → 35%  ✓
floor  → 28%  ✓
sofa   → 18%  ✓
roof   → 10%  ✓
table  → 6%   ✓
quantum → 0.1% ← cut off
```

Prevents the model from ever picking a very low probability token. Problem: K is fixed — doesn't adapt to the shape of the distribution.

---

## Top-P sampling (nucleus sampling)

Take the smallest set of tokens whose combined probability exceeds P. Adapts to the distribution shape.

```
P = 0.9:
mat    → 35%  cumulative: 35%  ✓
floor  → 28%  cumulative: 63%  ✓
sofa   → 18%  cumulative: 81%  ✓
roof   → 10%  cumulative: 91%  ✓ ← stop, passed 90%
table  → 6%   ← excluded
```

When the model is confident (one token at 95%) — tiny nucleus, maybe 1 token.
When the model is uncertain (many tokens equally likely) — large nucleus, many tokens.

**Top-P is generally preferred over Top-K** because it adapts. Most production systems use Top-P.

---

## Temperature + Top-P together

Commonly used together — but Anthropic and OpenAI recommend using one or the other, not both:

```typescript
// Recommended: temperature only
{ temperature: 0.7 }

// Recommended: top_p only
{ top_p: 0.9 }

// Not recommended
{ temperature: 0.7, top_p: 0.9 }
```

---

## Greedy decoding

Always pick the highest probability token. Temperature = 0 effectively. Deterministic and fast but produces repetitive or looping output. Not used much for open-ended generation.

---

## Beam search

Keep track of multiple candidate sequences simultaneously (beams). At each step, expand each beam and keep the top N overall sequences. Pick the highest probability sequence at the end.

Better than greedy at finding good sequences. Tends to produce safe, generic output. Used in translation and summarization — less so in open-ended LLM generation.

---

## Repetition penalty

Reduces the probability of tokens that have already appeared in the output. Prevents looping. Applied on top of whatever sampling strategy you're using.

---

## Min-P sampling

Sets a minimum probability threshold relative to the top token:

```
Top token: 40%
Min-P = 0.1 → only include tokens with probability ≥ 4% (10% of 40%)
```

Adapts the cutoff based on how confident the model is. Newer technique, gaining popularity.

---

## Anthropic API — how to set these

```typescript
const response = await anthropic.messages.create({
  model: "claude-opus-4-6",
  max_tokens: 1024,
  temperature: 0.7,
  top_p: 0.9,
  top_k: 40,
  messages: [
    { role: "user", content: "Your message here" }
  ]
});
```

**What Anthropic exposes:** temperature, top_p, top_k, max_tokens.
**What Anthropic does not expose:** repetition penalty (handled internally), beam search, min-p.

---

## max_tokens

The maximum number of tokens the model is allowed to generate per response. A hard cap — the model stops mid-sentence if it hits the limit.

```
Context window  → how many tokens the model can SEE as input
max_tokens      → how many tokens the model can WRITE as output
```

Each API call is independent. max_tokens resets every call — no memory of previous responses.

Output tokens typically cost more than input tokens on most APIs — generating is more expensive than reading.

**Practical values:**

```
Classification / structured output  → 256–512
Conversational response             → 1024
Detailed explanation                → 2048
Code generation                     → 4096
Long documents / reports            → 8192+
```

---

## Practical presets

```typescript
// Deterministic — structured output, JSON, data extraction
{ temperature: 0, max_tokens: 512 }

// Balanced — chatbot, Q&A, summarization
{ temperature: 0.7, max_tokens: 1024 }

// Creative — story generation, brainstorming
{ temperature: 1.0, top_k: 50, max_tokens: 2048 }

// Code generation
{ temperature: 0.2, max_tokens: 4096 }
```

For production systems returning structured data — set temperature to 0 and validate the output. Don't fight randomness when you need reliability.

---

## The key insight

All of these are post-training decisions. They don't change the model's weights or knowledge. They only change how the model picks from what it already calculated.

```
Model's job    → produce a probability distribution
Engineer's job → decide how to sample from it
```

---

## Key vocabulary

| Term | Meaning |
|---|---|
| Sampling | The process of picking a token from the probability distribution |
| Temperature | Controls how sharply or flatly the distribution is applied when sampling |
| Top-K | Only sample from the K most probable tokens |
| Top-P (nucleus) | Only sample from tokens whose cumulative probability exceeds P |
| Greedy decoding | Always pick the highest probability token — deterministic |
| Beam search | Explore multiple candidate sequences in parallel, pick the best |
| Repetition penalty | Reduces probability of tokens already generated — prevents loops |
| Min-P | Minimum probability threshold relative to the top token |
| max_tokens | Maximum number of tokens the model can generate per response |
| Logits | Raw scores before softmax — temperature operates on these |
