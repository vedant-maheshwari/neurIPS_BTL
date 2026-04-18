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

There are **two experiments** in this repository:

| Experiment | Script | Prompts | Layers | Purpose |
|------------|--------|---------|--------|---------|
| **Multilayer Sweep** | `svd_multilayer_sweep.py` | 1 control prompt (13 tokens) | L15–20 + L31 | Characterise spectral structure per layer; establish baseline |
| **BTL Sweep** | `svd_btl_sweep.py` | 42 BTL CoT prompts (~76 tokens) | L15–20 + L31 | Evaluate how cognitive complexity affects sensitivity to rank ablation |

---

## Table of Contents
1. [The Experiment — What We Did and How](#the-experiment)
2. [Metadata & Equations](#metadata--equations)
3. [Experiment 1: Multilayer Sweep — Plots & Inferences](#experiment-1-multilayer-sweep)
4. [Experiment 2: BTL Sweep — Plots & Inferences](#experiment-2-btl-sweep)
5. [Key Findings](#key-findings)
6. [Repository Structure](#repository-structure)
7. [Reading the Data](#reading-the-data)

---

## The Experiment

### The Model

**Phi-2 (microsoft/phi-2)** — 2.7B parameters, 32 layers, 32 heads, head dimension 80, full Multi-Head Attention.  
Loaded in `bfloat16` on GPU with `attn_implementation="eager"` to expose raw attention tensors.

### The Intervention: SVD Truncation

For each attention head independently, during forward pass:

Let `A ∈ ℝ^(Q×K)` be the causal post-softmax attention matrix. We decompose:

```
A = U Σ Vᵀ
```

And reconstruct a rank-k approximation:

$$A_k = \sum_{i=1}^{k} \sigma_i \, u_i v_i^\top = U_k \Sigma_k V_k^\top$$

`A_k` replaces `A` in `attn_output = A_k @ V_states`. All other computation is untouched. KV caching is **disabled** so truncation applies at every autoregressive step. Sweep: `k ∈ {1, 2, 3, 4, 5}`. Target layers: `{15, 16, 17, 18, 19, 20, 31}`.

### The BTL Prompts

42 Chain-of-Thought prompts (7 per level) across Bloom's six-level cognitive hierarchy:

| Level | Cognitive Demand | Example |
|-------|-----------------|---------|
| 1 — Remembering | Recall facts | "List the US Presidents of the 20th century in order." |
| 2 — Understanding | Explain, interpret | "Explain how a four-stroke internal combustion engine works." |
| 3 — Applying | Use knowledge to solve | "Calculate projectile trajectory at 45°, 50 m/s." |
| 4 — Analyzing | Decompose, compare | "Analyze thematic differences between Marvel and DC." |
| 5 — Evaluating | Judge, weigh evidence | "Evaluate the effectiveness of a four-day workweek." |
| 6 — Creating | Design, synthesise | "Design a public transportation system for a mountainous city." |

---

## Metadata & Equations

### Spectral Metadata (from the *original*, pre-truncation attention tensor)

These are properties of `A` — the model's natural attention patterns before any intervention.

| Metric | Formula | Intuition |
|--------|---------|-----------|
| **Singular values** | `SVD(A) → σ₁ ≥ σ₂ ≥ ...` | How attention energy is distributed across orthogonal modes |
| **Energy retained** | `E_ret(k) = (Σᵢ₌₁ᵏ σᵢ²) / (Σⱼ σⱼ²) × 100` | % of information the rank-k approximation preserves |
| **Spectral entropy** | `H = −Σᵢ pᵢ ln(pᵢ)`, `pᵢ = σᵢ²/Σσⱼ²` | High → many modes matter equally; Low → one mode dominates |
| **Effective rank** | `r_eff = exp(H)` | Continuous dimensionality: r_eff=1 means rank-1; r_eff=5 means ~5 active modes |
| **Top-1 dominance** | `d₁ = σ₁ / Σσᵢ` | Fraction of singular-value mass captured by the dominant mode |
| **Spectral gap** | `Δσ = σ₁ − σ₂` | Large gap → σ₁ truly dominates; Small gap → σ₁ and σ₂ compete |
| **# Significant SVs** | Count `σᵢ > 0.01·σ₁` | How many modes are "meaningfully sized" |

### Output Divergence Metrics (truncated vs. baseline)

| Metric | Formula | Intuition |
|--------|---------|-----------|
| **KL Divergence** | `D_KL(P_base ‖ P_k) = Σᵥ P_base(v) log(P_base(v)/P_k(v))` | How different are the next-token distributions |
| **Mean logit diff** | `(1/|V|) Σᵥ |L_base(v) − L_k(v)|` | Average logit perturbation across all vocabulary |
| **Max logit diff** | `max_v |L_base(v) − L_k(v)|` | Worst-case single-token perturbation |
| **Top-1 match** | Is argmax unchanged? | Did the most likely next token survive? |
| **Generation identity** | Token-for-token match over 200 tokens | Did the full output stay identical? |
| **Jaccard similarity** | `|W_base ∩ W_k| / |W_base ∪ W_k|` | Word-bag content overlap |
| **Shared prefix %** | Tokens identical before first divergence | How far into the response before a change |

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

## Key Findings

### 1. The Depth Gradient — Layer 31 is the Bottleneck

| Layer | r_eff | Entropy (bits) | Energy at k=1 | Top-1 Dom |
|-------|-------|----------------|---------------|-----------|
| L15 | 2.76 | 1.39 | 71.8% | 0.372 |
| L16 | 2.66 | 1.28 | 73.1% | **0.405** |
| L17 | 2.63 | 1.35 | 72.0% | 0.371 |
| L18 | 2.90 | 1.41 | 70.4% | 0.372 |
| L19 | 3.13 | 1.50 | 69.5% | 0.355 |
| L20 | 3.65 | 1.66 | 66.9% | 0.337 |
| **L31** | **5.17** | **2.01** | **50.1%** | **0.313** |

Layer 31 has 1.8× higher effective rank, 2.3× smaller spectral gap, and retains only half its energy at k=1. **It is the only layer that responds to cognitive complexity.** k=2 is sufficient for L15–L20 (≥98% energy); L31 requires k≥5.

### 2. The W-Shape — Cognitive Complexity ≠ Fragility

| BTL Level | KL (k=1) | Why |
|-----------|----------|-----|
| **Understanding** | **0.283** | Discourse register coherence requires modes 2–4 |
| **Applying** | 0.264 | Step scaffolding around math lives in secondary modes |
| **Creating** | 0.234 | Cross-domain constraint binding needs modes 4–5 |
| Remembering | 0.198 | Relative-clause qualifiers in modes 2–3 |
| Analyzing | 0.194 | Binary categorical anchors to mode 1 |
| **Evaluating** | **0.186** | Strong verdict tokens concentrate into σ₁ |

### 3. The Five Generation Archetypes

| Archetype | BTL Levels | Effect at k=1 |
|-----------|-----------|---------------|
| Formulaic Immunity | Applying (math) | Token-for-token identical output |
| Template Preservation | Remembering | Step scaffold preserved; qualifiers dropped |
| Lexical Substitution | Analyzing | Logic preserved; synonym choices change |
| Discourse Framework Shift | Understanding | Same facts, different rhetorical register |
| Interrogative Collapse | Evaluating | Imperative plans → exploratory questions; +7.7 words |
| Contextual Genericization | Creating | Domain constraints dropped; generic templates |

### 4. The Chaotic Trajectory Principle

```
r(r_eff(L31), D_KL)          = −0.006   (zero)
r(r_eff(L31), prefix_match%) = −0.106   (zero)
```

Spectral rank has **no predictive power** over text divergence. Small distributional perturbations compound over 200 autoregressive steps into completely different trajectories — a hallmark of chaotic dynamical systems.

---

## Repository Structure

```
neurIPS_BTL/
├── README.md                           ← This file
├── REPORT.md                           ← Technical summary (legacy)
├── btl_svd_analysis.md                 ← Full BTL × SVD spectral + KL analysis
├── btl_response_patterns.md            ← Qualitative text analysis + archetypes
├── svd_multilayer_analysis.md          ← Single-prompt multilayer sweep analysis
├── neurIPS_paper_FINAL.tex             ← Full 9-section NeurIPS paper (LaTeX)
├── section_generation_analysis.tex     ← Generation analysis section (LaTeX)
├── paper.tex                           ← Earlier draft
│
├── plots/                              ← BTL experiment plots (42 prompts × 7 BTL levels)
│   ├── kl_vs_btl.png                   ← KL Divergence vs BTL level (W-shape)
│   ├── heatmap_kl_btl_k.png            ← KL Divergence heatmap (BTL × k)
│   ├── eff_rank_vs_btl.png             ← Effective rank per layer × BTL level
│   ├── heatmap_rank_btl_layer.png      ← Effective rank heatmap (BTL × layer)
│   ├── energy_vs_btl.png               ← Energy retained at k=1 (per layer × BTL)
│   ├── entropy_vs_btl.png              ← Spectral entropy (per layer × BTL)
│   ├── dominance_vs_btl.png            ← Top-1 singular value dominance
│   ├── gen_match_vs_btl.png            ← Exact-match generation rates
│   ├── logit_diff_vs_btl.png           ← Mean logit difference vs BTL level
│   ├── seqlen_vs_rank.png              ← Sequence length vs L31 rank (control)
│   ├── eff_rank_vs_k_per_layer.png     ← Effective rank vs k (BTL-averaged)
│   ├── energy_vs_k_per_layer.png       ← Energy vs k (BTL-averaged)
│   │
│   └── multilayer/                     ← Multilayer sweep plots (1 control prompt)
│       ├── k_sweep_summary.png          ← KL/logit divergence across k values
│       ├── energy_vs_k_per_layer.png    ← Per-layer energy convergence vs k
│       ├── eff_rank_vs_k_per_layer.png  ← Per-layer effective rank (constant in k)
│       ├── k1/
│       │   ├── sv_spectra.png           ← Singular value spectra, all heads, all layers
│       │   ├── energy_retention.png     ← Per-head energy retained at k=1
│       │   ├── metadata_heatmaps.png    ← Rank / entropy / dominance / gap heatmaps
│       │   ├── attn_L15_H0.png          ← Attn matrix: orig vs truncated, Layer 15 Head 0
│       │   └── attn_L31_H0.png          ← Attn matrix: orig vs truncated, Layer 31 Head 0
│       ├── k3/
│       │   ├── metadata_heatmaps.png
│       │   └── attn_L31_H0.png
│       └── k5/
│           ├── metadata_heatmaps.png
│           └── attn_L31_H0.png
│
├── data/
│   └── all_results.json               ← 42-prompt × 5k × 7-layer experimental data
└── scripts/
    ├── svd_btl_sweep.py               ← BTL evaluation pipeline
    ├── svd_attention_intervention.py  ← Single-layer baseline patcher
    └── svd_multilayer_sweep.py        ← Multilayer spectral sweep
```

---

## Reading the Data (`data/all_results.json`)

```json
{
  "1_Remembering": [
    {
      "prompt": "List the US Presidents...",
      "prompt_id": "1_Remembering_P0",
      "seq_len": 76,
      "baseline": {
        "top5_tokens": [...],
        "top5_logits": [...],
        "generated": "Recall the first US President..."
      },
      "k_results": {
        "1": {
          "kl_divergence": 0.2124,
          "mean_logit_diff": 0.4512,
          "max_logit_diff": 2.1834,
          "top5_tokens": [...],
          "top1_match": true,
          "generated": "Recall the first US President...",
          "same_as_baseline": false,
          "layers": {
            "15": {
              "mean_effective_rank": 2.742,
              "mean_spectral_entropy": 1.378,
              "mean_energy_retained_pct": 71.9,
              "mean_top1_dominance": 0.3758,
              "mean_spectral_gap": 2.2511,
              "mean_num_significant_sv": 25.5,
              "min_energy_retained_pct": 61.2,
              "max_effective_rank": 4.1
            }
            // layers "16" through "31" follow same schema
          }
        }
        // "2", "3", "4", "5" follow same schema
      }
    }
    // 6 more prompts (P1–P6)
  ]
  // "2_Understanding" through "6_Creating" follow same schema
}
```

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
