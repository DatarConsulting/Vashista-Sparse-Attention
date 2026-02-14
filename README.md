# Vashista Sparse Attention — Reproducibility Notebook

This repository contains **a single, fully documented notebook** that accompanies the paper:

**_Attention in Constant Time: Vashista Sparse Attention for Long-Context Decoding with Exponential Guarantees_**  
Vashista Nobaub — Datar Consulting — labs@datar.fr

Code repository: https://github.com/DatarConsulting/Vashista-Sparse-Attention

---

## What the paper is about (plain English)

Long-context inference is expensive because standard attention must read and score **every token** in the context for each new query token.
In practice, however, attention often concentrates on a **small subset** of context tokens.

This paper makes that intuition **precise and testable**:

- It models “hard” attention as a **projection onto the convex hull** of key vectors.
- It models softmax attention as an **entropically regularized** relaxation of that projection.
- It proves a **face-stability / support-gap** result: when a *geometric margin* (the **support gap** \(\Delta\)) is positive,
  entropic attention concentrates on a **constant-size active set**, with **exponentially small leakage** onto irrelevant tokens.

This yields a principled story for *why* sparse attention can work and *when* it is safe to use.

---

## Why it matters (impact)

### 1) Predictable long-context cost (constant-in-\(T\))
In the certified regime (\(\Delta>0\)), the effective number of tokens that receive meaningful attention stays **bounded**
(up to exponentially small mass). That means per-token attention cost can be made **constant in the context length \(T\)**
using a small candidate set \(K_c\) and a small number of routed pages \(P\).

### 2) A certificate instead of a heuristic
Many sparse/streaming attention methods are heuristic: they work often, but failures can be hard to anticipate.
Here, the **KKT certificate** and the **support gap \(\Delta\)** give a concrete diagnostic for
when sparsification should be reliable and when a safe fallback should trigger.

### 3) Enterprise deployment: RAG, privacy, air-gapped inference
For RAG and other long-context enterprise use cases (multi-doc QA, compliance review, code/log copilots),
Vashista Sparse Attention supports:
- lower and more predictable latency and memory footprint,
- reduced context bandwidth costs,
- compatibility with **privacy-critical / air-gapped** deployments where data must not leave the environment,
- a drop-in pathway to interchangeable attention modules in inference stacks (e.g., vLLM / sglang).

---

## Contents (this repo is notebook-only)

- **`notebook_paper_repro_vashista_sparse_attention_documented.ipynb`**  
  End-to-end notebook aligned with the paper (Theory → Method → Experiments), designed to:
  - reproduce the tables/figures used in the manuscript (when the underlying cached outputs are available),
  - document assumptions, diagnostics, and failure modes (\(\Delta\approx 0\)),
  - provide reviewer-friendly mapping from paper claims → notebook cells → generated artifacts.

That’s it — **this repo contains only the notebook**.

---

## How to run

### Option A — Google Colab (recommended)
1. Open the notebook from GitHub in Colab (or upload it).
2. Run cells **top-to-bottom**.
3. The notebook installs dependencies and reproduces the paper outputs (tables/figures) when applicable.

### Option B — Local Jupyter
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip jupyter
jupyter lab
```
Open the notebook and run all cells.

---

## Reviewer notes (what to look for)

- **Support gap diagnostic**: how \(\widehat{\Delta}\) is estimated/proxied in practice.
- **Leakage behavior**: predicted off-face mass decay of the form \(\exp(-\Omega(\Delta/\varepsilon))\).
- **Dense regime**: what happens when \(\Delta\approx 0\) (fallback-to-dense vs capped compute policy).
- **Latency story**: why reducing memory reads over \(T\) dominates solver overhead for bounded \(K_c\).

---

## Contact

For questions or deployment discussions (privacy-critical / air-gapped inference):  
**labs@datar.fr**
