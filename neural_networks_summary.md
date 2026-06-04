# Neural Networks — Concept Summary

## What is a neural network?

A neural network is a function with a lot of adjustable knobs called **weights**. It learns the relationship between inputs and outputs by seeing many examples and being corrected repeatedly — nobody programs the rules by hand.

The key insight: instead of writing `if pointy ears → cat`, you show the network thousands of labeled images and let it figure out the rules itself.

---

## Layers and neurons

A network is organized in layers:

```
Input → [Layer 1] → [Layer 2] → [Layer 3] → Output
```

Each layer transforms the data a bit. Early layers learn simple patterns (edges, shapes), later layers combine those into complex ones (ears, fur texture). Each unit in a layer is a **neuron** — it multiplies its inputs by its weights, sums them up, and passes the result forward.

---

## Weights

Weights are the memory of the model. Each neuron multiplies its inputs by its own weights:

```
output = (x1 × w1) + (x2 × w2) + (x3 × w3)
```

- At the start of training, weights are **random** — the network knows nothing.
- After training, weights encode everything the model learned from the data.
- When people say a model has "70 billion parameters" — parameters is just another word for weights.

---

## How a neural network is created

Two distinct phases:

| Phase | Who | What |
|---|---|---|
| Design architecture | Researchers | Decide number of layers, neuron types, connections |
| Initialize weights | Computer | Set all weights to random values |
| Train | Computer + engineers | Feed data, adjust weights billions of times |
| Deploy | Engineers | Serve the trained model via API |
| Build on top | AI engineers (you) | RAG, agents, pipelines |

The architecture determines what the model is *capable* of learning. Training determines what it actually *knows*.

---

## The training process

### Batches and epochs

Training doesn't happen one image at a time. Images are fed in **batches** (typically 32–64 at a time):

```
Batch 1: images 1–32   → forward pass → measure error → adjust weights
Batch 2: images 33–64  → forward pass → measure error → adjust weights
...
```

One full pass through the entire dataset is called an **epoch**. Training runs for many epochs until accuracy stops improving.

### One training iteration (5 steps)

1. **Input** — raw data enters the network as numbers (e.g. pixel brightness values 0–1)
2. **Forward pass** — signal travels through every layer, each neuron computing its weighted sum
3. **Prediction** — output layer produces probabilities for each category (e.g. cat: 0.05, dog: 0.87)
4. **Error** — loss is calculated by comparing prediction to the correct label
5. **Adjust weights** — error travels backwards (backpropagation), every weight nudges slightly in the direction that reduces the loss

---

## How error is measured (loss)

The network doesn't output a word — it outputs **probabilities**:

```
cat  → 0.05
dog  → 0.87
bird → 0.06
car  → 0.02
```

The human label ("cat") gets automatically converted to a **one-hot vector**:

```
cat  → 1.0
dog  → 0.0
bird → 0.0
car  → 0.0
```

The **loss function** measures the gap between these two distributions. A common formula used is cross-entropy loss. The entire goal of training is to minimize this number:

```
Epoch 1:  loss = 2.31
Epoch 10: loss = 0.82
Epoch 50: loss = 0.09  ← good model
```

---

## Who does what in supervised learning

| Task | Who |
|---|---|
| Collect training data | Engineers |
| Label each example | Humans (just category names, no probabilities) |
| Convert labels to numbers | Code, automatically |
| Produce output probabilities | Network + softmax function, automatically |
| Calculate loss | Code, automatically |
| Adjust weights | Code, automatically |

Humans only assign simple labels ("cat", "dog"). The probability format, loss calculation, and weight updates are all automated.

---

## Key vocabulary

| Term | Meaning |
|---|---|
| Weight / parameter | A knob in the network that gets adjusted during training |
| Layer | A group of neurons that transforms data before passing it on |
| Forward pass | Running input data through the network to get a prediction |
| Backpropagation | Sending the error signal backwards to adjust weights |
| Loss | A single number measuring how wrong the prediction was |
| Epoch | One full pass through the entire training dataset |
| Batch | A small subset of training data processed together |
| One-hot vector | A label encoded as all zeros except a single 1 |
| Softmax | A function that converts raw outputs into probabilities (0–1, sum to 1) |
| Overfitting | When a model memorizes training data but fails on new examples |
