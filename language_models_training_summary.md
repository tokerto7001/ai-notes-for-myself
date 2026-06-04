# Language Models — Training Summary

## The core task: predict the next token

Unlike image classifiers where the label is a category ("cat", "dog"), language models are trained on a single objective:

> Given everything before this token, what comes next?

```
Input: "The cat sat on the"
Label: "mat"
```

This is called **self-supervised learning**. The label is already inside the text itself — no human labeling required.

---

## Every token is a training example

A single sentence generates multiple training examples automatically:

```
Input: "The"                  → Label: "cat"
Input: "The cat"              → Label: "sat"
Input: "The cat sat"          → Label: "on"
Input: "The cat sat on"       → Label: "the"
Input: "The cat sat on the"   → Label: "mat"
```

One sentence with 6 tokens = 5 training examples. Across trillions of tokens from the internet, this produces an almost infinite supply of labeled training data — for free.

---

## Where training data comes from

The internet. Literally.

- Common Crawl (web snapshots)
- Books and Wikipedia
- GitHub code
- Scientific papers, news, forums

Nobody labeled any of it. The structure of language itself provides the supervision.

---

## Error still exists — same mechanics as image classification

The model predicts probabilities for every possible next token:

```
mat    → 35%
floor  → 28%
sofa   → 18%
roof   → 10%
quantum → 0.1%
```

Loss measures how much probability the model assigned to the *correct* token — not just whether it picked the exact right word. Multiple continuations can be valid, so the model learns a distribution of likely next tokens.

Early in training:
```
Input: "The cat sat on the"
Prediction: "quantum"  ← nonsense, high loss
```

After training on trillions of tokens:
```
Input: "The cat sat on the"
Prediction: "mat" / "floor" / "sofa"  ← all reasonable, low loss
```

---

## Temperature uses the probability distribution

When you set temperature in an AI tool, you're controlling how the model samples from its output probabilities:

| Temperature | Effect | Good for |
|---|---|---|
| 0 | Always pick the highest probability token. Deterministic. | Code, factual Q&A |
| 1 (default) | Sample according to probabilities as they are. Natural variation. | Most use cases |
| < 1 | Sharpens distribution — top tokens dominate more | Consistency with some variation |
| > 1 | Flattens distribution — low probability tokens become more likely | Creative writing, brainstorming |

Temperature doesn't change what the model knows. It only changes how the model picks from what it already calculated. High temperature causes hallucinations because it pulls from low-probability tokens the model isn't confident about.

---

## Pretraining alone isn't enough

After pretraining, the model is a text completion machine — not an assistant. It continues text in whatever style it sees, but doesn't know how to answer questions helpfully.

The next phase — **fine-tuning / RLHF** — shapes it into a useful assistant. (Covered in topic 8.)

---

## Training vs inference — the neural network runs in both

A common misconception: the neural network only runs during training. It runs at inference too.

| | Training | Inference |
|---|---|---|
| Forward pass | ✓ | ✓ |
| Loss calculation | ✓ | ✗ |
| Weight adjustment | ✓ | ✗ |
| Weights change | Yes | No |
| When it happens | Once, before deployment | Every time you use the model |

Every time you send a message to an AI, the full neural network runs — all the layers, all the weights, all the math — to produce each token of the response. Weights are frozen at inference, which is why correcting the model in a conversation doesn't permanently change it.

---

## Why next-token prediction produces intelligence

To predict the next token well, the model must implicitly learn:

- Grammar and syntax
- Facts about the world
- Cause and effect
- Tone and style
- Logic and reasoning

The task is simple. What's required to do it well is not.

---

## Scale

| | Image classifier | Large language model |
|---|---|---|
| Training data | Millions of labeled images | Trillions of tokens |
| Parameters | Millions | Billions to trillions |
| Hardware | Few GPUs | Thousands of GPUs |
| Training time | Hours to days | Months |
| Cost | Thousands of dollars | Hundreds of millions of dollars |

Same fundamental mechanics. Incomprehensibly different scale.

---

## Key vocabulary

| Term | Meaning |
|---|---|
| Self-supervised learning | Training where labels come from the data itself, no human annotation needed |
| Token | A unit of text the model processes (roughly ¾ of a word on average) |
| Pretraining | The initial large-scale training on internet text |
| Inference | Running the model to get a response — no weight updates, just forward pass |
| Temperature | Controls how strictly the model follows its probability distribution when picking tokens |
| Hallucination | When the model generates confident but wrong output — often caused by sampling from low-probability tokens |
| Fine-tuning | Further training after pretraining to shape the model's behavior |
