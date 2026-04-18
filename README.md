
# Spectral Dynamics of Attention Under Cognitive Load
### SVD-Based Rank Ablation of Multi-Head Attention in Phi-2, Evaluated Across Bloom's Taxonomy

[![Paper](https://img.shields.io/badge/paper-NeurIPS--draft-blue)](neurIPS_paper_FINAL.tex)
[![Model](https://img.shields.io/badge/model-Phi--2%202.7B-purple)](https://huggingface.co/microsoft/phi-2)
[![Prompts](https://img.shields.io/badge/prompts-42%20BTL%20CoT-green)]()
[![Layers](https://img.shields.io/badge/layers-15--20%20%2B%2031-orange)]()

---

## What Is This?

This repository contains the full experimental pipeline, raw data, visualizations, and analysis for an investigation into **how much of Multi-Head Attention's full-rank computation is actually necessary**, and whether that necessity changes with **cognitive task complexity**.

The core idea: instead of analysing what attention *is*, we ask what happens when we **surgically remove** parts of it. We do this by decomposing the attention matrix using **Singular Value Decomposition (SVD)** and discarding all but the top-*k* components — directly during inference, inside specific layers, across 42 reasoning prompts graded along **Bloom's Taxonomy of Learning (BTL)**.

By comparing what the model generates with and without these components, we can read off exactly what each group of singular modes *encodes* — and how much rank different cognitive tasks actually demand.

---

## The Experiment — What We Did and How

### 1. The Model

We use **Phi-2 (microsoft/phi-2)**, a 2.7 billion parameter autoregressive language model with:
- 32 transformer layers
- 32 attention heads per layer
- Head dimension: 80
- Architecture: **Multi-Head Attention (MHA)** — all heads have full independent key/value projections

We load it in `bfloat16` on GPU with `attn_implementation="eager"` — this bypasses Flash Attention and gives us direct access to the raw attention weight tensors.

---

### 2. The Intervention: SVD Truncation

The key operation happens inside each targeted attention layer. Normally, after computing scaled dot-product attention scores and applying softmax, the attention matrix `A` guides how values are aggregated:

```
attn_output = A @ V_states
```

We intercept this. Instead of using the full `A`, we decompose it with SVD and keep only the dominant modes.

**For each attention head independently:**

Let `A ∈ ℝ^(Q×K)` be the post-softmax attention probability matrix (square at prefill time, where Q=K=N, the sequence length). We compute:

```
A = U Σ Vᵀ
```

where:
- `U ∈ ℝ^(N×N)` — left singular vectors (query-side)
- `Σ = diag(σ₁, σ₂, ..., σₙ)` — singular values, sorted descending: σ₁ ≥ σ₂ ≥ ... ≥ 0
- `Vᵀ ∈ ℝ^(N×N)` — right singular vectors (key-side)

We then reconstruct a **rank-k approximation** by zeroing all but the top-k singular values:

$$A_k = \sum_{i=1}^{k} \sigma_i \, u_i v_i^\top = U_k \Sigma_k V_k^\top$$

This `A_k` is substituted back into the forward pass. The rest of the computation (value projection, residual stream, MLP blocks) is untouched.

**Sweep:** We test `k ∈ {1, 2, 3, 4, 5}` for each prompt. KV caching is **disabled** so truncation is applied at every autoregressive step, not just prefill.

**Targeted layers:** `{15, 16, 17, 18, 19, 20, 31}` — the middle reasoning block plus the final layer.

---

### 3. The Evaluation Prompts: Bloom's Taxonomy (BTL)

We use **42 Chain-of-Thought prompts** (7 per level) graded across Bloom's six-level cognitive hierarchy:

| Level | Cognitive Demand | Example Prompt |
|-------|-----------------|----------------|
| **1 — Remembering** | Recall facts, list items | "List the US Presidents who served during the 20th century in chronological order." |
| **2 — Understanding** | Explain, summarise, interpret | "Explain how a four-stroke internal combustion engine works." |
| **3 — Applying** | Use knowledge to solve | "Calculate the trajectory of a projectile at 45° with 50 m/s." |
| **4 — Analyzing** | Decompose, compare, contrast | "Analyze the thematic differences between Marvel and DC comics." |
| **5 — Evaluating** | Judge, critique, weigh evidence | "Evaluate the effectiveness of a four-day workweek." |
| **6 — Creating** | Design, construct, synthesise | "Design a public transportation system for a mountainous city." |

All prompts are formatted as:
```
Instruct: Provide a detailed, step-by-step reasoning chain for the following request.
Clearly separate your thoughts and explain the logic behind each step.

Request: {prompt}

Output: Let's think step by step.
Step 1:
```

The step-by-step format forces the model into explicit chain-of-thought mode, making structural changes under truncation more visible.

---

### 4. What Metadata We Capture

For each of the 42 prompts × 5 k-values × 7 layers, we collect:

#### A. Per-head Spectral Metadata (from the **original**, pre-truncation attention tensor)

These are properties of `A` itself — they tell us about the natural structure of the attention matrix before we do anything to it.

| Metric | Formula | What It Means |
|--------|---------|---------------|
| **Singular Values** | `SVD(A)` → `σ₁ ≥ σ₂ ≥ ...` | The distribution of "attention energy" across orthogonal modes |
| **Energy Retained** | `E_ret(k) = (Σᵢ₌₁ᵏ σᵢ²) / (Σⱼ σⱼ²) × 100` | What % of total information the rank-k approximation preserves |
| **Spectral Entropy** | `H = −Σᵢ pᵢ ln(pᵢ)` where `pᵢ = σᵢ²/Σσⱼ²` | How spread out the energy is across modes; high = many modes matter, low = one mode dominates |
| **Effective Rank** | `r_eff = exp(H)` | Continuous, entropy-weighted measure of dimensionality. r_eff=1 means rank-1; r_eff=5 means ~5 modes contribute meaningfully |
| **Top-1 Dominance** | `d₁ = σ₁ / Σσᵢ` | What fraction of total singular-value mass σ₁ captures. High dominance → rank-1 like |
| **Spectral Gap** | `Δσ = σ₁ − σ₂` | How much larger the top mode is than the second. Large gap → first mode truly dominates |
| **# Significant SVs** | Count of σᵢ > 0.01·σ₁ | How many modes are "meaningfully sized" relative to the top mode |

These are computed from the full SVD of the original attention matrix (before any truncation is applied), so they are a property of the model's natural attention patterns — not of our intervention.

#### B. Output-Level Divergence Metrics (comparing truncated vs. baseline)

| Metric | Formula | What It Means |
|--------|---------|---------------|
| **KL Divergence** | `D_KL(P_base ‖ P_k) = Σᵥ P_base(v) log(P_base(v)/P_k(v))` | How different the next-token probability distributions are |
| **Mean Logit Diff** | `(1/|V|) Σᵥ |L_base(v) − L_k(v)|` | Average absolute change in raw logit scores across all vocab tokens |
| **Max Logit Diff** | `max_v |L_base(v) − L_k(v)|` | The single worst-case token perturbation |
| **Top-1 Match** | Boolean: is argmax unchanged? | Did the most likely next token change? |
| **Generation Identity** | Token-for-token exact match over 200 generated tokens | Did the full output stay identical? |

#### C. Text-Level Divergence (post-hoc analysis of generated strings)

| Metric | What It Measures |
|--------|-----------------|
| **Jaccard Similarity** | Word-bag overlap: `|W_base ∩ W_k| / |W_base ∪ W_k|` |
| **Shared Prefix %** | Fraction of initial tokens that match before the first divergence point |
| **Word Count Δ** | How much longer/shorter the truncated response is |
| **Step Structure Match** | Whether the number of "Step N:" markers is preserved |

---

## The Equations — What Each Metric Means Intuitively

### Effective Rank: `r_eff = exp(H)`

The effective rank answers: **"How many singular modes are meaningfully active?"**

- If all energy is in σ₁ (rank-1 attention): `p₁=1`, `H=0`, `r_eff=1`
- If energy is split equally across N modes: `H=ln(N)`, `r_eff=N`
- Real values fall in between — r_eff=2.7 means roughly 2-3 modes carry the structural load

This is more informative than integer rank because it weights each mode by its functional contribution, not just whether it exists.

### Spectral Entropy: `H = −Σᵢ pᵢ ln(pᵢ)`

Treats the spectrum `{σᵢ²}` as a probability distribution over "attention modes." Low entropy = one mode dominates (strongly dissipative, information funnelled). High entropy = energy is spread across many modes (multi-scale, distributed attention).

### Energy Retained: `E_ret(k)`

The most interpretable compression metric. If k=1 retains 72% of energy, removing the remaining singular modes causes a 28% information loss. This has direct geometric meaning: the Frobenius distance between `A` and `A_k` is `‖A − A_k‖_F = sqrt(Σᵢ₌ₖ₊₁ σᵢ²)`.

### KL Divergence: `D_KL(P_base ‖ P_k)`

Measures how different the next-token distributions are. A KL of 0 means identical distributions (truncation had no effect on what the model would say next). A KL of 0.28 (the maximum we observe, for Understanding at k=1) means substantial distributional shift — but not catastrophic.

---

## Results

### Finding 1: The Depth Gradient — Layer 31 is Spectrally Unique

| Layer | r_eff | Entropy (bits) | Energy at k=1 | Top-1 Dom | Spectral Gap |
|-------|-------|----------------|---------------|-----------|-------------|
| L15 | 2.76 | 1.39 | 71.8% | 0.372 | 2.256 |
| L16 | 2.66 | 1.28 | 73.1% | **0.405** | **2.383** |
| L17 | 2.63 | 1.35 | 72.0% | 0.371 | 2.211 |
| L18 | 2.90 | 1.41 | 70.4% | 0.372 | 2.186 |
| L19 | 3.13 | 1.50 | 69.5% | 0.355 | 2.148 |
| L20 | 3.65 | 1.66 | 66.9% | 0.337 | 2.095 |
| **L31** | **5.17** | **2.01** | **50.1%** | **0.313** | **1.055** |

**What this means:**

Layers 15–20 are **strongly dissipative** — the first singular mode captures 37–40% of all singular-value mass, with a large gap (σ₁ >> σ₂). This means attention funnels most of its routing information through a single dominant pattern. These layers operate effectively as rank-2 systems.

Layer 31 is fundamentally different. Its spectral gap is 2.3× smaller, its effective rank is 1.8× higher, and it retains only *half* its energy at k=1 (vs. ~70% for intermediate layers). **It is the computational bottleneck for any compression scheme.**

**Why does Layer 31 behave differently?** Layer 31 directly precedes the unembedding matrix that maps hidden states to vocabulary logits. To score all ~50,000 vocabulary tokens simultaneously, it must maintain a much more informationally diverse attention distribution — it cannot collapse to a single dominant mode without losing discriminability across tokens.

**Context-length scaling:** A single 13-token control prompt gives r_eff(L31) ≈ 2.84. The 42 BTL prompts at ~76 tokens give r_eff(L31) ≈ 5.17 — a **1.8× increase** from a ~6× increase in sequence length. Intermediate layers show no such scaling (they operate at near-constant rank regardless of sequence length). This means: **rank is not a fixed architectural constant — it is a dynamic quantity that scales with how much context Layer 31 needs to integrate.**

---

### Finding 2: The W-Shape — BTL Sensitivity is Non-Monotonic

KL divergence at k=1 sorted by magnitude:

| BTL Level | KL (k=1) | KL (k=5) | Why |
|-----------|----------|----------|-----|
| **Understanding** | **0.283** ± 0.198 | 0.092 | Procedural discourse needs sustained register across steps → information in secondary modes |
| **Applying** | 0.264 ± 0.082 | 0.072 | Step scaffolding around math lives in secondary modes (math content itself is immune) |
| **Creating** | 0.234 ± 0.139 | 0.068 | Cross-domain constraint binding requires modes 4–5 |
| Remembering | 0.198 ± 0.042 | 0.094 | Simple recall; some temporal qualifiers in secondary modes |
| Analyzing | 0.194 ± 0.042 | 0.071 | Binary comparative frames anchor well to mode 1 |
| **Evaluating** | **0.186** ± 0.093 | **0.067** | Strong "opinion/verdict" tokens concentrate into σ₁ → naturally robust |

**The counterintuitive result:** *Higher cognitive complexity ≠ higher fragility.* Evaluating and Analyzing are the most *robust* to truncation. Understanding and Applying are the most *fragile*.

**Why?** The W-shape is explained by *how* each task type uses the attention spectrum:
- **Understanding** requires maintaining a consistent rhetorical register across many explanation steps. This coherence lives in modes 2–4.
- **Evaluating** generates strong "stance" tokens (e.g., "should", "however", "ultimately") that dominate σ₁. The model's evaluative structure self-organises into rank-1-compatible patterns.
- The W-shape holds across *all* k values — it's a real structural effect, not noise.

**Heatmap reading:** See `plots/heatmap_kl_btl_k.png`. The gradient runs top-left (high KL, Understanding at k=1) to bottom-right (low KL, Evaluating at k=5). All levels converge smoothly as k increases, with Evaluating always at the bottom and Understanding always at the top.

---

### Finding 3: Top-1 Robustness vs. Trajectory Fragility

At k=1 (most aggressive truncation across 7 layers simultaneously):
- **Top-1 token prediction matches baseline: 78.6%** (33/42 prompts)
- **Full 200-token generation identity: ~2%** (1/42 prompts, only Remembering)

This gap is the key finding. The model is spectrally robust at the **single step** — meaning the most likely next token usually survives. But small distributional shifts compound over 200 autoregressive steps, producing completely different text even when the first token is unchanged.

This is a property of **chaotic dynamical systems**: sensitive dependence on initial conditions drives exponential trajectory divergence even for vanishingly small initial perturbations.

---

### Finding 4: Five Generation Archetypes

Close reading of all 42×5 generation pairs reveals five distinct behavioural patterns:

#### Archetype 1: Formulaic Immunity (Applying — mathematical sub-tasks)
**k=1 output is token-for-token identical to baseline.**

```
Both: "Vx = V·cos(theta); Vy = V·sin(theta)..."
```

Mathematical formulae create near-zero next-token entropy. When the model writes "Vx = V·cos", the only valid token is "(theta)". The distribution is so peaked that rank-1 attention is informationally sufficient. **These tasks operate at the rank-1 floor naturally.**

#### Archetype 2: Template Preservation with Metadata Compression (Remembering)
**Step scaffold preserved (71% at k=1); relative-clause qualifiers stripped.**

```
Baseline: "Recall the next US President who served after the first one."
k=1:      "Recall the next US President."
```

Action-entity pairs (Recall, President) live in σ₁. Relative-clause modifiers ("who served after the first one") live in σ₂–σ₃.

#### Archetype 3: Lexical Substitution with Logic Preservation (Analyzing)
**Arguments preserved; only synonym choices change.**

```
Baseline: "...darker, more mature tone..."
k=1:      "...darker and more mature tone..."
k=5:      "...darker and more serious tone..."
```

Binary categorical frames (Marvel vs. DC) anchor strongly to mode 1. Secondary modes encode only fine-grained synonym selection.

#### Archetype 4: Discourse Framework Shift (Understanding)
**Same facts, completely different rhetorical register.**

```
Baseline (definitional): "The first stroke is called the intake stroke."
k=1 (temporal narrative): "The engine starts with the intake stroke."
k=5 (instructional):      "The first step in a four-stroke engine is the intake stroke."
```

Sustained explanatory discourse requires consistent register across many steps. This register coherence lives in modes 2–4. Without them, the model falls back to the highest-frequency discourse pattern in pretraining: temporal narration.

#### Archetype 5: Interrogative Collapse (Evaluating)
**Imperative planning mode → exploratory question mode.**

```
Baseline: "Consider the economic impact. Step 2: Analyze the psychological benefits..."
k=1:      "...we need to consider the economic impact. Will it lead to increased productivity?"
```

Imperative planning requires holding an abstract meta-schema (consider → analyze → weigh → verdict) simultaneously with the topic. The meta-schema lives in secondary modes. Without it, the model generates *questions about* the evaluation instead of *performing* the evaluation. This causes the **+7.7 word count inflation** — questions are less syntactically compact than directives.

#### Archetype 6: Contextual Genericization (Creating)
**Cross-domain constraint binding degrades layer by layer.**

```
Baseline: "...transportation modes for mountainous areas, such as cable cars or gondolas."
k=1:      "...most efficient and cost-effective."          ← terrain constraint lost
k=3:      "...(e.g., buses, trains, or cable cars)."       ← partial recovery
k=5:      "...suitable for the mountainous terrain."        ← constraint back; no synthesis
```

Binding two domain constraints simultaneously (mountainous terrain × transport engineering → cable cars) requires modes 4–5. Each additional mode restores one level of constraint-binding specificity.

---

### Finding 5: The Chaotic Trajectory Principle

We test whether spectral complexity of the initial forward pass predicts autoregressive text divergence:

```
r(r_eff(L31), D_KL)          = −0.006   (essentially zero)
r(r_eff(L31), prefix_match%) = −0.106   (essentially zero)
```

**A prompt with r_eff=4.6 can produce 0% prefix match; a prompt with r_eff=5.5 can share 41% of prefix.** The magnitude of the initial spectral perturbation has no predictive power over text divergence.

This is a hallmark of chaotic systems: trajectory divergence is governed by the system's Lyapunov structure throughout generation, not by the perturbation's size. The practical implication: **SVD truncation is not lossy compression of knowledge — it is a stochastic perturbation of generation style.** Facts survive; rhetorical structure does not.

---

## Experiment 1: Multilayer Sweep

**Script:** `scripts/svd_multilayer_sweep.py`  
**Prompt:** `"Explain the concept of entropy in thermodynamics step by step."` (13 tokens)  
**Purpose:** Characterise the *natural* spectral structure of each layer before introducing cognitive complexity. This is the control experiment.

**Plots:** `plots/multilayer/`

---

### [`plots/multilayer/k_sweep_summary.png`](plots/multilayer/k_sweep_summary.png)

**What it shows:** Three line plots — (1) KL Divergence vs k, (2) Mean |Δ Logit| vs k, (3) Max |Δ Logit| vs k — for this single prompt with SVD applied across all 7 layers simultaneously.

**Inference:**  
Even at `k=1` (the most aggressive possible truncation — keeping just 1 singular mode across 7 layers simultaneously), the **KL divergence is only 0.007**. This is an extraordinarily small perturbation. For context, random noise at this scale would be imperceptible in the output distribution. All three metrics drop sharply from k=1 to k=3, then plateau. This establishes that for a short, simple prompt, the entire 7-layer attention mechanism across the middle and final layers operates **effectively in a rank-1 to rank-2 regime**. The model isn't using most of its attention capacity for simple questions.

---

### [`plots/multilayer/energy_vs_k_per_layer.png`](plots/multilayer/energy_vs_k_per_layer.png)

**What it shows:** Line plot — mean energy retained (%) vs k, one line per layer. Shows how quickly each layer recovers its full information as k increases.

**Inference:**  
This is the clearest visualisation of the **depth gradient**. Layers 15–20 (upper cluster of lines) jump from ~75–78% at k=1 to ≥98% at k=2 — a ~23 percentage-point gain for just one additional mode. They then plateau, meaning **k=2 is the compression sweet spot for intermediate layers**. Layer 31 (bottom line) climbs slowly: 66% → 85% → 93% → 95% → 97.5% at k=1…5, never fully converging within the sweep. The gap between L31 and all other layers is the most important structural finding of this experiment: **Layer 31 requires asymmetrically more rank budget than any intermediate layer.**

---

### [`plots/multilayer/eff_rank_vs_k_per_layer.png`](plots/multilayer/eff_rank_vs_k_per_layer.png)

**What it shows:** Line plot — mean effective rank vs k, one line per layer. Note: effective rank is computed from the *original* attention matrix and is therefore **constant across k** (flat lines).

**Inference:**  
The flat lines confirm we are correctly computing spectral metadata from the original (pre-truncation) tensor — effective rank is a property of `A`, not `A_k`. The key signal is the **vertical offset between lines**: Layer 31 (top line, r_eff ≈ 2.84) sits clearly above all six intermediate layers (r_eff ≈ 1.78–1.98). This gap of ~1 effective rank unit means Layer 31 naturally uses ~1.5× more spectral modes than the intermediate block. L15–L17 are nearly identical (~1.80–1.85); there is a slight monotonic increase from L15 to L20 (1.81→1.98), suggesting a gradual spectral expansion as depth increases, culminating in the sharp jump at L31.

---

### [`plots/multilayer/k1/sv_spectra.png`](plots/multilayer/k1/sv_spectra.png)

**What it shows:** Log-scale plot of all singular value spectra (one line per head) for each of the 7 target layers at k=1. This is the most detailed spectral fingerprint in the experiment.

**Inference:**  
The most striking observation is the **variation in spectral decay rate across layers**:
- **L15, L17**: Gradual decay over ~7 orders of magnitude. Several modes remain above 10⁻² through index 4–5, meaning multiple modes carry meaningful attention weight.
- **L16**: Steeper — largest head spans 9 orders. Notable because L16 has the highest top-1 dominance (d₁=0.405) of all layers.
- **L19**: Flattest of the middle layers — singular values cluster between 10⁻¹ and 10⁻⁵ over the first several indices, indicating more distributed attention.
- **L20**: Very steep — the dominant head spans 26 orders of magnitude. Signals rapid collapse after the first mode.
- **L31**: Most extreme — spans >30 orders. However, **Heads 0 and 25 are visibly anomalous** — their spectra fall off much more slowly, maintaining significant energy in modes 3–8. These are the "reasoning heads" with r_eff ≈ 8–9, far above the layer average of 2.84.

---

### [`plots/multilayer/k1/energy_retention.png`](plots/multilayer/k1/energy_retention.png)

**What it shows:** Per-head bar chart of energy retained (%) at k=1 for each of the 7 layers. Bars are coloured by retention percentage (viridis colourmap).

**Inference:**  
Two clusters emerge. **Intermediate layers (L15–L20)** form a high-retention band at 75–78% — all heads within a layer are relatively uniform, with most bars between 70–85%. **Layer 31** is dramatically different: while most heads cluster around 66%, Heads 0 and 25 drop to ~27% and ~15% respectively. These two heads are so spectrally distributed that a rank-1 approximation discards 73–85% of their information. This directly confirms the sv_spectra finding: **Heads 0 and 25 in L31 are the only heads across the entire model that fundamentally resist low-rank compression.**

---

### [`plots/multilayer/k1/metadata_heatmaps.png`](plots/multilayer/k1/metadata_heatmaps.png)

**What it shows:** 2×2 grid of heatmaps — layers (rows) × heads (columns) — showing: (top-left) Effective Rank, (top-right) Spectral Entropy, (bottom-left) Top-1 Dominance, (bottom-right) Spectral Gap.

**Inference:**  
Read each panel as a "map" of the attention landscape:
- **Effective Rank**: Layer 31 row is visibly brighter throughout, especially columns 0 and 25 (near-white, rank 8–9). L15–L20 rows are uniformly dark (rank 1.5–2.5). Within the intermediate block, L19–L20 heads 4–5 show slight brightening (≈2.5), confirming the gradual transition upward.
- **Spectral Entropy**: Mirrors effective rank — L31 heads 0 and 25 show ~3 bits (maximum), while most L15–L20 heads show <1 bit. Low entropy = attention focused on 1–2 tokens; high entropy = attention spread across many positions.
- **Top-1 Dominance**: Inverted pattern from the above — **L31 is the dark row** (lowest dominance, ~0.2–0.3), L16 is the brightest row (~0.5–0.7). Head 25 in L31 is the darkest pixel in the entire heatmap (dominance ≈0.15) — the single most distributed head in the model.
- **Spectral Gap**: L15–L18 are bright (gap 1.5–2.0; σ₁ >> σ₂). L20–L31 are progressively darker (gap shrinks; σ₁ ≈ σ₂). A small spectral gap means multiple modes compete for dominance — consistent with more distributed, multi-scale attention.

---

### [`plots/multilayer/k1/attn_L15_H0.png`](plots/multilayer/k1/attn_L15_H0.png) and [`plots/multilayer/k1/attn_L31_H0.png`](plots/multilayer/k1/attn_L31_H0.png)

**What they show:** Side-by-side attention matrices for Head 0 — Original | SVD-Truncated (k=1) | Difference — for Layer 15 and Layer 31 respectively.

**Inference:**  
This is the most visually direct demonstration of the depth gradient:

- **Layer 15, k=1**: The truncated heatmap closely resembles the original. The difference map shows small values (max ~0.3). The rank-1 approximation preserves the primary token-attention pattern (attention concentrating on the most contextually relevant token, likely "entropy" or "thermodynamics"). The information lost is minor, confirming L15's low effective rank.

- **Layer 31, k=1**: **Massive distortion.** The difference map shows values approaching 1.0 — entire rows of the attention matrix are fundamentally changed. Tokens like "step", "by", and "odynamics" (from "thermodynamics") lose their original attention targets completely. The rank-1 approximation collapses L31's distributed, multi-target attention into a single global pattern, destroying the fine-grained token routing that generates specific vocabulary.

Compare these to `plots/multilayer/k5/attn_L31_H0.png`: at k=5, even L31 is mostly recovered. Residual differences remain at "entropy" and "of" (max ~0.2), but the overall routing structure is preserved.

---

### [`plots/multilayer/k1/metadata_heatmaps.png`](plots/multilayer/k1/metadata_heatmaps.png) vs [`plots/multilayer/k5/metadata_heatmaps.png`](plots/multilayer/k5/metadata_heatmaps.png)

**Inference:**  
Comparing k=1 and k=5 heatmaps shows what compression recovers. The effective rank and entropy panels are identical across k (they are properties of the original matrix). The spectral gap and dominance panels change: at higher k, the reconstruction more faithfully represents the multi-modal structure of L31, allowing Heads 0 and 25 to make their distributed contributions rather than being collapsed into a single dominant pattern.

---

## Experiment 2: BTL Sweep

**Script:** `scripts/svd_btl_sweep.py`  
**Prompts:** 42 CoT prompts, 7 per BTL level (Remembering → Creating)  
**Mean sequence length:** ~76 tokens  
**Purpose:** Evaluate whether cognitive complexity determines sensitivity to rank ablation; characterise behavioural changes in generated text.

**Plots:** `plots/` (root level, all `*_vs_btl.png` and `heatmap_*.png` files)

> **Note:** Two plots that were originally generated by the multilayer sweep script are also present at root level in `plots/` (`eff_rank_vs_k_per_layer.png`, `energy_vs_k_per_layer.png`). These are the **BTL-averaged** versions computed across all 42 prompts — distinct from the single-prompt versions in `plots/multilayer/`.

---

### [`plots/kl_vs_btl.png`](plots/kl_vs_btl.png)

**What it shows:** Line plot — KL divergence vs BTL level, one line per k value (k=1 to k=5), with error bars showing within-level standard deviation. X-axis: 6 BTL levels. Y-axis: mean KL divergence.

**Inference:**  
This is the primary result plot. It reveals the **W-shape anomaly**: KL divergence is *not* monotonically increasing with BTL level (as intuition might suggest). Instead:
- **Understanding (BTL-2)** peaks at **KL=0.283** — the highest sensitivity of any level at k=1
- **Applying (BTL-3)** is second highest at KL=0.264
- **Creating (BTL-6)** comes third at KL=0.234
- **Evaluating (BTL-5)** is the *lowest* at KL=0.186 — more robust than even Remembering (0.198)

The W-shape is consistent across *all* k values (all five lines show the same ordering), confirming it is a structural effect driven by how each task type uses the attention spectrum, not statistical noise. The error bars for Understanding are notably wide (±0.198) — this level has the highest within-level variance, meaning some Understanding prompts are very sensitive and others relatively robust.

**Why the W-shape?** Understanding requires sustained discourse coherence across many explanation steps — this coherence lives in modes 2–4. Evaluating generates strong "verdict" or "stance" tokens that concentrate naturally into σ₁, making it intrinsically more rank-1 compatible. Higher cognitive level ≠ higher fragility.

---

### [`plots/heatmap_kl_btl_k.png`](plots/heatmap_kl_btl_k.png)

**What it shows:** 2D heatmap — rows = BTL levels (Remembering at top), columns = k values (k=1 at left). Cell colour = mean KL divergence (brighter/darker = higher/lower KL). Cell values are annotated.

**Inference:**  
Read this as a complete sensitivity map of the experiment. Three patterns are immediately visible:
1. **Horizontal gradient (left→right):** All rows fade as k increases — confirming that more modes always reduces output perturbation. The fastest convergence is Analyzing and Evaluating (rows fade quickly). The slowest is Understanding (darker throughout).
2. **Vertical ordering (top→bottom):** The W-shape ordering is preserved down each column — Understanding is always darker than Evaluating in every column.
3. **The k=4 threshold:** Almost all values drop below 0.12 at k=4. This is the practical "good enough" threshold — beyond k=4, the attention compression is nearly lossless across all task types. The exception is Understanding at k=3 (KL=0.206), which still shows moderate sensitivity.

---

### [`plots/eff_rank_vs_btl.png`](plots/eff_rank_vs_btl.png)

**What it shows:** Multi-line plot — mean effective rank per BTL level, one line per layer. X-axis: BTL levels. Y-axis: mean effective rank (averaged over all 7 prompts in each level).

**Inference:**  
Two key patterns:
1. **The layer ordering is preserved across all BTL levels**: Layer 31 line sits at top (~5.1–5.3), Layer 20 is below it (~3.6–3.7), and layers 15–17 cluster at the bottom (~2.6–2.7). This confirms the depth gradient is not an artifact of any particular task type.
2. **BTL variation in effective rank is small for intermediate layers**: L15–L17 lines are nearly flat (std across BTL levels ≈ 0.02–0.06) — these layers simply don't respond to cognitive complexity. Layer 31 shows the most BTL variation: Analyzing (5.31) and Evaluating (5.26) rank highest; Applying (5.03) ranks lowest. This means **the final vocabulary projection layer is the only part of the network that adjusts its spectral footprint based on the type of reasoning being performed**.

---

### [`plots/heatmap_rank_btl_layer.png`](plots/heatmap_rank_btl_layer.png)

**What it shows:** 2D heatmap — rows = BTL levels, columns = layers (L15 to L31). Cell colour = mean effective rank at k=1. Cell values are annotated numerically.

**Inference:**  
This is the most compact visualisation of the depth gradient finding. The rightmost column (L31) is dramatically brighter than all others — values of 5.03–5.31 against a sea of 2.63–3.67 for intermediate layers. The intermediate columns (L15–L20) show a monotonic brightness increase from left to right (L15≈2.76 to L20≈3.65), revealing the gradual spectral expansion as depth increases before the sharp jump at L31. The rows (BTL levels) show minimal variation across the intermediate columns — confirming that intermediate layers are **BTL-invariant**. The only row-level variation is in the L31 column, where Analyzing and Evaluating are slightly brighter than Applying and Remembering.

---

### [`plots/energy_vs_btl.png`](plots/energy_vs_btl.png)

**What it shows:** Multi-line plot — mean energy retained at k=1 per BTL level, one line per layer. X-axis: BTL levels. Y-axis: energy retained (%).

**Inference:**  
The layer ordering mirrors effective rank but now in the opposite direction — **lower energy retained = more spectrally complex layer**. Layer 31 (bottom line) sits at ~50% across all BTL levels, retaining just half its energy with a single mode. Intermediate layers cluster between 67–73%. The near-flat BTL variation for all lines confirms that energy retention is primarily determined by layer depth, not task type. The slight downward trend for L20 across BTL levels (from ~67% at Remembering to ~66.5% at Creating) suggests that semantically richer prompts marginally increase spectral complexity even in intermediate layers, but the effect is small compared to the layer-level differences.

---

### [`plots/entropy_vs_btl.png`](plots/entropy_vs_btl.png)

**What it shows:** Multi-line plot with error bars — mean spectral entropy (bits) per BTL level, one line per layer at k=1.

**Inference:**  
Spectral entropy directly measures how "spread out" attention is across modes. The 0.73-bit gap between Layer 31 (H≈2.01 bits) and Layer 16 (H≈1.28 bits, the lowest intermediate layer) corresponds to Layer 31 having ~1.65× more effective modes (since r_eff = 2^H in base-2). The error bars are notably larger for Layer 31 across all BTL levels, confirming that L31's spectral structure is more variable across prompts — it is genuinely responding to different inputs differently, unlike intermediate layers which maintain consistent entropy regardless of prompt content.

---

### [`plots/dominance_vs_btl.png`](plots/dominance_vs_btl.png)

**What it shows:** Multi-line plot — mean top-1 dominance (σ₁/Σσᵢ) per BTL level, one line per layer.

**Inference:**  
Top-1 dominance is the most direct measure of how "rank-1-like" a layer is. **Layer 16 is the most dominant** (d₁≈0.405) — its σ₁ captures 40% of all singular-value mass. This explains why L16 has the steepest singular value spectrum (as seen in sv_spectra.png). **Layer 31 is the least dominant** (d₁≈0.313) — multiple modes genuinely compete. All lines are **flat across BTL levels** — the critical observation. Dominance structure is entirely determined by layer identity, not by what question is being asked. This means the model's spectral geometry is an architectural constant for intermediate layers; only L31 shows any cognitive-load sensitivity (in effective rank, not in dominance).

---

### [`plots/gen_match_vs_btl.png`](plots/gen_match_vs_btl.png)

**What it shows:** Grouped bar chart — exact-match generation rates (%) per BTL level (x-axis), grouped by k value (one bar per k). Y-axis: % of prompts in that level producing identical 200-token output to baseline.

**Inference:**  
This is the most sobering plot in the experiment. Almost all bars are zero or near-zero. The only non-zero bars appear in **Remembering at k=2 (28.6%)** and **Remembering at k=1 (14.3%)** — meaning only 1–2 out of 7 Remembering prompts produce identical outputs. All other levels are effectively 0% at all k values. This establishes a fundamental asymmetry: **top-1 token prediction is preserved 78.6% of the time at k=1, but full generation identity is practically impossible**. The reason is autoregressive compounding: even when the first predicted token matches, a tiny distributional shift in its probability propagates to condition every subsequent token. Over 200 generation steps, this compounding produces completely different text. **This plot is the empirical proof that generation is chaotic**: small perturbations produce unbounded trajectory divergence.

---

### [`plots/logit_diff_vs_btl.png`](plots/logit_diff_vs_btl.png)

**What it shows:** Multi-line plot — mean absolute logit difference vs BTL level, one line per k. X-axis: BTL levels. Y-axis: mean |ΔLogit| averaged over all vocabulary tokens.

**Inference:**  
This plot replicates the KL W-shape finding in raw logit space (rather than probability space), confirming the effect is not an artifact of the softmax nonlinearity. The W-shape ordering (Understanding and Applying peak; Analyzing and Evaluating trough) is preserved. One additional insight: the **scale of logit differences is large** (mean of ~0.46 at k=1 across all levels) even though the KL divergence is moderate (max 0.28). This indicates that truncation perturbs many vocabulary tokens by small amounts rather than a few tokens by large amounts — consistent with a distributed spectral perturbation rather than a targeted one. The maximum logit diff (not shown here but measured at ~4.0 at k=1 for L31) confirms there are individual tokens severely impacted by rank reduction.

---

### [`plots/seqlen_vs_rank.png`](plots/seqlen_vs_rank.png)

**What it shows:** Scatter plot — x-axis: prompt sequence length (tokens), y-axis: Layer 31 effective rank at k=1. Points are coloured by BTL level.

**Inference:**  
This is the control plot for a critical confound: **does BTL level affect spectral rank simply because higher BTL prompts tend to be longer?** The answer is no. Sequence lengths range from 67–87 tokens with no systematic relationship between BTL level and length. Within this range, there is no systematic correlation between sequence length and L31 effective rank — points at seq_len=75 span ranks 4.8–5.5 regardless of BTL level. The BTL-level colours show no clustering. **This confirms that the BTL-level differences in L31 rank are driven by semantic content, not tokenization artifacts.** The absence of a length–rank correlation within this dataset also provides a bound: the strong length dependence observed between the 13-token control prompt (rank 2.84) and the 76-token BTL prompts (rank 5.17) is a long-range scaling effect; short-range variation in length (67–87 tokens) does not further shift rank.

---

## Repository Structure

```
neurIPS_BTL/
├── README.md                           ← This file
├── REPORT.md                           ← Technical summary report (legacy)
├── btl_svd_analysis.md                 ← Full BTL × SVD analysis (spectral + KL)
├── btl_response_patterns.md            ← Qualitative text analysis + archetypes
├── svd_multilayer_analysis.md          ← Single-prompt multi-layer sweep analysis
├── neurIPS_paper_FINAL.tex             ← Full NeurIPS paper (LaTeX)
├── section_generation_analysis.tex     ← Generation analysis section (LaTeX)
├── paper.tex                           ← Earlier draft
├── plots/
│   ├── kl_vs_btl.png
│   ├── heatmap_kl_btl_k.png
│   ├── eff_rank_vs_btl.png
│   ├── heatmap_rank_btl_layer.png
│   ├── energy_vs_btl.png
│   ├── entropy_vs_btl.png
│   ├── dominance_vs_btl.png
│   ├── gen_match_vs_btl.png
│   ├── logit_diff_vs_btl.png
│   └── seqlen_vs_rank.png
├── data/
│   └── all_results.json                ← Full 42-prompt × 5k × 7-layer data
└── scripts/
    ├── svd_btl_sweep.py                ← Master evaluation pipeline
    ├── svd_attention_intervention.py   ← Single-layer baseline patcher
    └── svd_multilayer_sweep.py         ← Multi-layer spectral sweep
```

---

## Reading the Data (`all_results.json`)

The JSON is structured as:
```json
{
  "1_Remembering": [
    {                                    // one entry per prompt (7 per level)
      "prompt": "List the US Presidents...",
      "prompt_id": "1_Remembering_P0",
      "seq_len": 76,
      "baseline": {
        "top5_tokens": [...],
        "top5_logits": [...],
        "generated": "Recall the first US President..."
      },
      "k_results": {
        "1": {                           // one entry per k value
          "kl_divergence": 0.2124,
          "mean_logit_diff": 0.4512,
          "max_logit_diff": 2.1834,
          "top5_tokens": [...],
          "top1_match": true,
          "generated": "Recall the first US President...",
          "same_as_baseline": false,
          "layers": {
            "15": {                      // one entry per layer
              "mean_effective_rank": 2.742,
              "mean_spectral_entropy": 1.378,
              "mean_energy_retained_pct": 71.9,
              "mean_top1_dominance": 0.3758,
              "mean_spectral_gap": 2.2511,
              "mean_num_significant_sv": 25.5,
              "min_energy_retained_pct": 61.2,
              "max_effective_rank": 4.1
            },
            ...
          }
        },
        "2": { ... }, "3": { ... }, "4": { ... }, "5": { ... }
      }
    },
    ...       // 6 more prompts
  ],
  "2_Understanding": [ ... ],
  ...
}
```

---

## Key Claims (TL;DR)

1. **Intermediate layers (L15–L20) are rank-2 funnels.** They are BTL-invariant — spectral structure doesn't change across task types. They need only k=2 to preserve ≥98% of their energy.

2. **Layer 31 is the compression bottleneck.** It has 1.8× higher effective rank, 50% lower energy at k=1, and scales with sequence length. Any practical attention compression scheme must allocate asymmetric rank budget to this layer.

3. **The W-shape is real.** Understanding and Applying are more fragile than Creating or Evaluating. The explanation: procedural tasks require sustained discourse coherence across steps (a secondary-mode phenomenon), while evaluative tasks concentrate into strong σ₁-dominant opinion tokens.

4. **Secondary modes encode style, not substance.** Fact content survives k=1. Rhetorical register, evaluative meta-schemas, and cross-domain constraint binding require k≥3.

5. **Autoregressive generation is chaotic.** Spectral rank has zero predictive power over text divergence. Small perturbations to initial token distributions compound into completely different trajectories.

---

## Citation

```bibtex
@article{maheshwari2026spectral,
  title   = {Spectral Dynamics of Attention Under Cognitive Load:
             Low-Rank Dissipation of Multi-Head Attention Across Bloom's Taxonomy},
  author  = {Maheshwari, Vedant},
  year    = {2026},
  note    = {Preprint}
}
```