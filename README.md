# Spectral Attention Analysis Over Bloom's Taxonomy 🧠
*Evaluating the dissipation and generation fidelity of Phi-2 (MHA) under active cognitive load.*

This repository holds the raw testing data, visualizations, and a drafted paper reporting on the low-rank properties of Multi-Head Attention when subjected to reasoning prompts derived from Bloom's Taxonomy of Learning (BTL). 

We utilize a low-rank Singular Value Decomposition (SVD) projection pipeline to dynamically intervene on attention calculation midway through the causal pass.

## 📖 Main Draft
A detailed layout of the theory, SVD truncation equations, mathematical formulas for Effective Rank, and extensive plotting interpretations is available here:
**[Read the Full Report (REPORT.md)](REPORT.md)**

## 📂 Repository Structure

- `scripts/`: Source code tracking generating the results.
  - `svd_btl_sweep.py`: The master evaluation script that patches SVD truncation into layers 15-20 and 31.
  - `svd_attention_intervention.py`: Baseline patcher.
  - `svd_multilayer_sweep.py`: Standalone SVD property testing.
- `plots/`: SVD intervention heatmap clusters, degradation plots, and structural energy analysis. View visual results in the [report](REPORT.md).
- `data/`: `all_results.json` containing the massive extracted trace of KL-divergence, Effective Rank, and top-$k$ predictions across 42 taxonomy tasks.

## 🚀 Key Takeaways
1. Multi-Head Attention acts as a **low-rank dissipator**. The bulk of structural energy sinks into the primary singular value, specifically towards the end of the transformer backbone (e.g. Layer 31 $r_{eff} \approx 1.5$).
2. Higher cognitive constraints ("Evaluating", "Creating") demand access to the long-tail singular modes. Truncating representation below $k=4$ completely shatters generation reasoning.
3. Because simple "Remembering" tasks hold up nicely beneath extreme dimensionality reduction, attention implementations can theoretically utilize an **adaptive-rank inference** gateway, modulating dimensional rank based on incoming request complexity.
