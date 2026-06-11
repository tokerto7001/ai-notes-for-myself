# RLHF & Alignment — Summary

## The problem with a raw pretrained model

After pretraining, the model has one skill: continue text. It doesn't know it's supposed to answer questions, follow instructions, or decline harmful requests. It just continues patterns from the internet.

Ask it "What is the capital of France?" and it might generate more questions — because questions appear in lists on the internet.

**Pretraining gives capability. It doesn't give alignment with human intent.**

---

## The three stage pipeline

```
Pretraining → SFT → Reward Model → PPO (RLHF)
```

---

## Stage 1 — Supervised Fine Tuning (SFT)

Human contractors write thousands of examples of ideal conversations:

```
User: What is the capital of France?
Assistant: The capital of France is Paris.

User: How do I make a bomb?
Assistant: I can't help with that.
```

The pretrained model is fine-tuned on these examples — same training mechanics, just on a small curated dataset instead of the entire internet.

After SFT the model starts behaving like an assistant. But it's limited by the quality and coverage of the examples written.

---

## Stage 2 — Train a Reward Model

Instead of writing more perfect examples, human raters now compare outputs. The model generates multiple responses to the same prompt and humans rank them:

```
Prompt: "Explain black holes simply"

Response A: technical explanation
Response B: analogy-based explanation
Response C: vague explanation

Human ranking: B > A > C
```

Ranking is much easier than writing perfect answers. These rankings train a separate neural network — the **reward model** — which learns to predict human preference. Given a response, it outputs a score representing how good a human would rate it.

---

## Stage 3 — PPO (Reinforcement Learning)

The reward model is used to improve the main LLM:

```
LLM generates response
        ↓
Reward model scores it
        ↓
High score → reinforce this behavior
Low score  → move away from this behavior
        ↓
LLM weights update
        ↓
Repeat thousands of times
```

The algorithm used is **PPO — Proximal Policy Optimization**. It nudges the model toward responses humans prefer without drifting too far from the original pretrained model in any single step.

This full pipeline — reward model + reinforcement learning — is called **RLHF: Reinforcement Learning from Human Feedback**.

---

## What RLHF actually changes

| Before RLHF | After RLHF |
|---|---|
| Optimizes for: predict the next token | Optimizes for: produce responses humans prefer |
| Text completion engine | Useful assistant |
| Answers everything or nothing | Declines harmful requests appropriately |
| Confident even when wrong | Admits uncertainty |
| No consistent persona | Consistent tone and behavior |

---

## Constitutional AI — Anthropic's approach

Anthropic developed a variation called **Constitutional AI (CAI)**. Instead of relying entirely on human raters:

1. Define a set of principles — a **constitution** (e.g. "respect human autonomy", "be honest")
2. Model generates a response
3. Another AI critiques the response against the constitution
4. Response is revised based on the critique
5. Revised responses become training data

This reduces reliance on human raters for the feedback signal — scaling the alignment process. **Claude is trained with CAI.**

---

## The full picture — from raw model to deployed assistant

```
Internet text (trillions of tokens)
        ↓
Pretraining — learns language, knowledge, patterns
        ↓
SFT — learns to behave like an assistant
        ↓
Reward model training — learns human preferences
        ↓
RLHF / CAI — aligns behavior with human values
        ↓
Claude / GPT / Gemini etc.
```

The raw pretrained model is never what gets deployed.

---

## RLHF limitations

**Reward hacking** — the model learns to game the reward model rather than genuinely improve. Finds responses that score high but aren't actually better.

**Reward model errors** — the reward model is itself imperfect. If it misjudges quality, the LLM optimizes toward those errors.

**Human rater disagreement** — different people prefer different things. Averaging across raters loses nuance.

**Sycophancy** — the model can learn to tell people what they want to hear rather than what's true, because agreement tends to score well with human raters.

Alignment research is still a very active field. RLHF is a significant improvement over raw pretraining but far from solved.

---

## Key vocabulary

| Term | Meaning |
|---|---|
| Alignment | Making model behavior consistent with human values and intent |
| SFT (Supervised Fine Tuning) | Fine-tuning on curated human-written examples of ideal behavior |
| Reward model | A separate neural network trained to predict human preference scores |
| RLHF | Reinforcement Learning from Human Feedback — using a reward model to improve the LLM |
| PPO | Proximal Policy Optimization — the RL algorithm used to update the LLM's weights |
| Constitutional AI (CAI) | Anthropic's approach — using a written constitution and AI critique instead of human raters |
| Reward hacking | When the model games the reward model rather than genuinely improving |
| Sycophancy | Model tendency to agree with users and say what they want to hear rather than what's true |
