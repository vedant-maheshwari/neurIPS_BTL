"""
SVD Attention Intervention × BTL CoT Prompts — Phi-2 (Kaggle Edition)
=====================================================================
Runs SVD truncation (k=1-5) across layers 15-20 + 31 for all 42 BTL prompts.
Collects per-layer spectral metadata and compares across Bloom's Taxonomy levels.
"""

import os, gc, json, time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "./svd_btl_results"
TARGET_LAYERS = [15, 16, 17, 18, 19, 20, -1]
K_VALUES = [1, 2, 3, 4, 5]
MAX_NEW_TOKENS = 200       # shorter than full gen to save time across 42 prompts
RUN_GENERATION = True      # set False to skip text generation (saves ~30 min)
SAVE_INCREMENTALLY = True  # save after each prompt for crash recovery

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "data"), exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# BTL PROMPTS (6 levels × 7 prompts = 42 total)
# ═══════════════════════════════════════════════════════════════════════════════
BTL_PROMPTS = {
    "1_Remembering": [
        "List the US Presidents who served during the 20th century in chronological order. Walk me through your memory retrieval step-by-step.",
        "State the three laws of motion formulated by Isaac Newton. Think step-by-step to define each one accurately.",
        "Recall the sequence of major events that led to the start of World War I. Outline the timeline step-by-step.",
        "List the main organs of the human digestive system. Step-by-step, trace the path of food from start to finish.",
        "Identify the primary layers of the Earth's atmosphere from lowest to highest. State each one step-by-step.",
        "Name the steps of the scientific method. Walk through the standard sequence step-by-step.",
        "List the five basic tastes detected by the human tongue. Break down your list step-by-step."
    ],
    "2_Understanding": [
        "Explain how a four-stroke internal combustion engine works. Think step-by-step to describe what happens in each stroke.",
        "Summarize the plot of Romeo and Juliet. Break down the narrative arc step-by-step so a beginner can understand.",
        "Explain the process of photosynthesis. Detail the inputs, mechanisms, and outputs step-by-step.",
        "Describe the concept of inflation in economics. Think step-by-step using a simple analogy involving everyday goods.",
        "Explain how a transformer neural network differs from a recurrent neural network. Break down the structural differences step-by-step.",
        "Clarify the greenhouse effect. Explain step-by-step how solar radiation interacts with Earth's atmosphere.",
        "Interpret the meaning of 'opportunity cost'. Walk through a real-world scenario step-by-step to illustrate it."
    ],
    "3_Applying": [
        "Calculate the trajectory of a projectile launched at 45 degrees with an initial velocity of 50 m/s. Show your work step-by-step.",
        "Apply the principles of supply and demand to predict what happens to ice cream prices during a heatwave. Walk through the logic step-by-step.",
        "Given a patient with a severe peanut allergy who accidentally ingested peanuts, outline the immediate first-aid response step-by-step.",
        "Demonstrate how to use the binary search algorithm to find the number 37 in a sorted list from 1 to 100. Write out the steps.",
        "Calculate the compound interest on a $10,000 principal at 5% annually over 5 years. Show every step of your math.",
        "Apply classical conditioning to write a training plan for a dog to sit on command. Detail the process step-by-step.",
        "Use the Pythagorean theorem to find the length of a ladder needed to reach a 12-foot window if the base is 5 feet away. Solve it step-by-step."
    ],
    "4_Analyzing": [
        "Analyze the thematic differences between Marvel and DC comics. Break down your analysis step-by-step, covering tone and world-building.",
        "Deconstruct the systemic causes of the 2008 financial crisis. Think step-by-step to connect the housing bubble to global markets.",
        "Compare and contrast the breathing mechanisms of fish and mammals. Analyze the biological differences step-by-step.",
        "Examine the socio-economic factors driving urban gentrification. Break down the causes and effects step-by-step.",
        "Analyze the structural differences between a Shakespearean sonnet and a Petrarchan sonnet step-by-step.",
        "Deconstruct a classic phishing email. Think step-by-step to analyze the psychological triggers used to deceive the reader.",
        "Investigate the relationship between rising ocean temperatures and hurricane intensity. Break down the thermodynamics step-by-step."
    ],
    "5_Evaluating": [
        "Evaluate the effectiveness of a four-day workweek. Think step-by-step to weigh the economic, psychological, and logistical factors before giving a final verdict.",
        "Critique the ethics of using facial recognition technology in public spaces. Weigh the pros and cons step-by-step to reach a conclusion.",
        "Assess whether nuclear energy is a viable solution to climate change. Provide a step-by-step justification of your final stance.",
        "Judge the historical success of the League of Nations. Argue step-by-step why it ultimately failed.",
        "Evaluate the claim that 'universal basic income reduces the incentive to work'. Provide a step-by-step defense or refutation based on economic theory.",
        "Critique GDP as a measure of a country's well-being. Step-by-step, evaluate its flaws and propose a better alternative.",
        "Determine which is a greater threat to modern cybersecurity: social engineering or malware. Justify your choice step-by-step."
    ],
    "6_Creating": [
        "Design a new public transportation system for a highly mountainous city. Construct your proposal step-by-step, addressing engineering and cost.",
        "Create a comprehensive plan for a human colony on Mars. Step-by-step, design the life support, governance, and daily routines.",
        "Invent a novel board game that teaches children about supply chain logistics. Write out the core rules and mechanics step-by-step.",
        "Formulate a new geopolitical treaty to govern the mining of asteroids. Draft the core tenets and enforcement mechanisms step-by-step.",
        "Design a sustainable, closed-loop waste management system for a skyscraper. Outline the architecture step-by-step.",
        "Compose a detailed outline for a science fiction novel where humanity loses the ability to lie. Build the world step-by-step.",
        "Create a hypothetical biological mechanism that would allow humans to safely digest plastic. Detail the enzymes and digestive steps."
    ]
}

# ═══════════════════════════════════════════════════════════════════════════════
# SVD CORE FUNCTIONS (same as working single-prompt version)
# ═══════════════════════════════════════════════════════════════════════════════
def svd_truncate_attention(attn_weights, top_k):
    B, H, Q, K = attn_weights.shape
    attn_f32 = attn_weights.float()
    reconstructed = torch.zeros_like(attn_f32)
    for b in range(B):
        for h in range(H):
            A = attn_f32[b, h]
            U, S_vals, Vt = torch.linalg.svd(A, full_matrices=False)
            k = min(top_k, S_vals.shape[0])
            S_vals[k:] = 0.0
            reconstructed[b, h] = U @ torch.diag(S_vals) @ Vt
    return reconstructed.to(attn_weights.dtype)


def compute_spectral_metadata(attn_tensor, top_k):
    """Per-head spectral metadata from attention tensor (B, H, Q, K)."""
    attn = attn_tensor[0].numpy()
    H = attn.shape[0]
    meta = {
        "singular_values_top5": [],
        "effective_rank": np.zeros(H),
        "spectral_entropy": np.zeros(H),
        "energy_retained_pct": np.zeros(H),
        "top1_dominance": np.zeros(H),
        "spectral_gap": np.zeros(H),
        "frobenius_norm": np.zeros(H),
        "num_significant_sv": np.zeros(H),  # SVs > 1% of σ₁
    }
    for h in range(H):
        sv = np.linalg.svd(attn[h], compute_uv=False)
        sv = sv[sv > 1e-30]
        meta["singular_values_top5"].append(sv[:5].tolist())
        energy = sv ** 2
        total = np.sum(energy)
        meta["frobenius_norm"][h] = np.sqrt(total)
        k = min(top_k, len(sv))
        meta["energy_retained_pct"][h] = (np.sum(energy[:k]) / total * 100) if total > 0 else 100.0
        if total > 0:
            p = energy / total; p = p[p > 0]
            meta["spectral_entropy"][h] = -np.sum(p * np.log2(p))
            ent_nats = -np.sum(p * np.log(p))
            meta["effective_rank"][h] = np.exp(ent_nats)
        else:
            meta["spectral_entropy"][h] = 0.0
            meta["effective_rank"][h] = 1.0
        meta["top1_dominance"][h] = sv[0] / np.sum(sv) if len(sv) > 0 else 1.0
        meta["spectral_gap"][h] = (sv[0] - sv[1]) if len(sv) > 1 else sv[0]
        meta["num_significant_sv"][h] = np.sum(sv > 0.01 * sv[0]) if len(sv) > 0 else 0
    return meta


def svd_eager_attention_forward(module, query, key, value, attention_mask, scaling,
                                 dropout=0.0, top_k=5, capture_dict=None, layer_id=None, **kwargs):
    from transformers.models.phi.modeling_phi import repeat_kv
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    original = attn_weights.clone()
    attn_weights = svd_truncate_attention(attn_weights, top_k=top_k)
    if capture_dict is not None and layer_id is not None:
        Q_len, K_len = original.shape[2], original.shape[3]
        if Q_len == K_len:
            capture_dict[layer_id] = {
                "original_attn": original.detach().cpu().float(),
                "truncated_attn": attn_weights.detach().cpu().float(),
            }
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class MultiLayerSVDPatcher:
    def __init__(self, model, top_k=5, layer_indices=None):
        self.model, self.top_k = model, top_k
        self.layer_indices = layer_indices or [-1]
        self.capture = {}
        self._originals = {}

    def __enter__(self):
        num_layers = len(self.model.model.layers)
        resolved = [i if i >= 0 else num_layers + i for i in self.layer_indices]
        for layer_idx in resolved:
            attn_mod = self.model.model.layers[layer_idx].self_attn
            self._originals[layer_idx] = attn_mod.forward
            top_k, capture = self.top_k, self.capture

            def make_fwd(am, lid):
                def fwd(hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kw):
                    from transformers.models.phi.modeling_phi import apply_rotary_pos_emb
                    shape = hidden_states.shape[:-1]
                    hshape = (*shape, -1, am.head_dim)
                    q = am.q_proj(hidden_states).view(hshape).transpose(1, 2)
                    k = am.k_proj(hidden_states).view(hshape).transpose(1, 2)
                    v = am.v_proj(hidden_states).view(hshape).transpose(1, 2)
                    if getattr(am, "qk_layernorm", False):
                        q, k = am.q_layernorm(q), am.k_layernorm(k)
                    cos, sin = position_embeddings
                    nd = am.rotary_ndims
                    qr, kr = apply_rotary_pos_emb(q[..., :nd], k[..., :nd], cos, sin)
                    q = torch.cat((qr, q[..., nd:]), dim=-1)
                    k = torch.cat((kr, k[..., nd:]), dim=-1)
                    if past_key_values is not None:
                        k, v = past_key_values.update(k, v, am.layer_idx)
                    out, w = svd_eager_attention_forward(am, q, k, v, attention_mask,
                        dropout=0.0, scaling=am.scaling, top_k=top_k, capture_dict=capture, layer_id=lid)
                    out = out.reshape(*shape, -1).contiguous()
                    out = getattr(am, "dense", getattr(am, "o_proj", None))(out)
                    return out, w
                return fwd
            attn_mod.forward = make_fwd(attn_mod, layer_idx)
        return self

    def __exit__(self, *a):
        for li, fwd in self._originals.items():
            self.model.model.layers[li].self_attn.forward = fwd
        self._originals.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE PROMPT PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
def run_single_prompt(model, tokenizer, prompt, resolved_layers):
    """Run baseline + k=1-5 SVD intervention for one prompt. Returns dict of results."""

    # Format prompt for Phi-2
    formatted = (
        f"Instruct: Provide a detailed, step-by-step reasoning chain for the following request. "
        f"Clearly separate your thoughts and explain the logic behind each step.\n\n"
        f"Request: {prompt}\n\nOutput: Let's think step by step.\nStep 1:"
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    result = {
        "prompt": prompt,
        "seq_len": seq_len,
        "baseline": {},
        "k_results": {},
    }

    # ── Baseline forward ──
    with torch.no_grad():
        baseline_out = model(**inputs)
    baseline_logits = baseline_out.logits
    baseline_next = baseline_logits[0, -1, :]
    baseline_probs = torch.softmax(baseline_next, dim=0)
    top5_base = torch.topk(baseline_next, 5)

    result["baseline"]["top5_tokens"] = [tokenizer.decode(i) for i in top5_base.indices.tolist()]
    result["baseline"]["top5_logits"] = [round(v, 4) for v in top5_base.values.tolist()]

    # ── Baseline generation ──
    if RUN_GENERATION:
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        prompt_len = input_ids.shape[1]
        result["baseline"]["generated"] = tokenizer.decode(gen[0][prompt_len:], skip_special_tokens=True)
        del gen
    else:
        result["baseline"]["generated"] = ""

    # ── SVD sweep ──
    for top_k in K_VALUES:
        k_data = {"top_k": top_k, "layers": {}}

        # Forward pass
        with torch.no_grad():
            with MultiLayerSVDPatcher(model, top_k=top_k, layer_indices=TARGET_LAYERS) as patcher:
                mod_out = model(**inputs)
                capture = dict(patcher.capture)

        mod_logits = mod_out.logits
        mod_next = mod_logits[0, -1, :]
        mod_probs = torch.softmax(mod_next, dim=0)
        mod_top5 = torch.topk(mod_next, 5)

        # Output metrics
        logit_diff = (baseline_logits - mod_logits).abs()
        kl = torch.sum(baseline_probs * (torch.log(baseline_probs + 1e-10) - torch.log(mod_probs + 1e-10))).item()

        k_data["mean_logit_diff"] = round(logit_diff.mean().item(), 6)
        k_data["max_logit_diff"] = round(logit_diff.max().item(), 6)
        k_data["kl_divergence"] = round(kl, 8)
        k_data["top5_tokens"] = [tokenizer.decode(i) for i in mod_top5.indices.tolist()]
        k_data["top5_logits"] = [round(v, 4) for v in mod_top5.values.tolist()]
        k_data["top1_match"] = (mod_top5.indices[0].item() == top5_base.indices[0].item())

        # Per-layer spectral metadata
        for layer_idx in resolved_layers:
            if layer_idx in capture:
                meta = compute_spectral_metadata(capture[layer_idx]["original_attn"], top_k)
                k_data["layers"][str(layer_idx)] = {
                    "mean_effective_rank": round(float(np.mean(meta["effective_rank"])), 4),
                    "mean_spectral_entropy": round(float(np.mean(meta["spectral_entropy"])), 4),
                    "mean_energy_retained_pct": round(float(np.mean(meta["energy_retained_pct"])), 2),
                    "mean_top1_dominance": round(float(np.mean(meta["top1_dominance"])), 4),
                    "mean_spectral_gap": round(float(np.mean(meta["spectral_gap"])), 4),
                    "mean_num_significant_sv": round(float(np.mean(meta["num_significant_sv"])), 2),
                    "min_energy_retained_pct": round(float(np.min(meta["energy_retained_pct"])), 2),
                    "max_effective_rank": round(float(np.max(meta["effective_rank"])), 4),
                }

        # Generation
        if RUN_GENERATION:
            with torch.no_grad():
                with MultiLayerSVDPatcher(model, top_k=top_k, layer_indices=TARGET_LAYERS):
                    gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                         do_sample=False, pad_token_id=tokenizer.eos_token_id)
            prompt_len = input_ids.shape[1]
            gen_text = tokenizer.decode(gen[0][prompt_len:], skip_special_tokens=True)
            k_data["generated"] = gen_text
            k_data["same_as_baseline"] = (gen_text == result["baseline"]["generated"])
            del gen
        else:
            k_data["generated"] = ""
            k_data["same_as_baseline"] = None

        result["k_results"][str(top_k)] = k_data

        # Cleanup
        del mod_out, mod_logits, mod_next, mod_probs, logit_diff, capture
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    del baseline_out, baseline_logits, baseline_next, baseline_probs
    gc.collect()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_btl_comparison(all_data, resolved_layers):
    """Generate aggregate comparison plots across BTL levels."""
    plot_dir = os.path.join(OUTPUT_DIR, "plots")
    btl_levels = list(BTL_PROMPTS.keys())
    btl_short = [b.split("_")[1] for b in btl_levels]
    colors = plt.cm.Set2(np.linspace(0, 1, len(K_VALUES)))

    # ── 1. KL Divergence vs BTL Level (per k) ──
    fig, ax = plt.subplots(figsize=(12, 6))
    for ki, k in enumerate(K_VALUES):
        means, stds = [], []
        for btl in btl_levels:
            kls = [p["k_results"][str(k)]["kl_divergence"] for p in all_data[btl]]
            means.append(np.mean(kls)); stds.append(np.std(kls))
        ax.errorbar(range(len(btl_levels)), means, yerr=stds, marker="o", label=f"k={k}",
                    color=colors[ki], linewidth=2, capsize=4)
    ax.set_xticks(range(len(btl_levels))); ax.set_xticklabels(btl_short, fontsize=11)
    ax.set_xlabel("Bloom's Taxonomy Level", fontsize=12); ax.set_ylabel("KL Divergence", fontsize=12)
    ax.set_title("KL Divergence vs BTL Level (SVD Truncation)", fontsize=14, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(plot_dir, "kl_vs_btl.png"), dpi=150); plt.show(); plt.close()

    # ── 2. Mean Logit Diff vs BTL Level (per k) ──
    fig, ax = plt.subplots(figsize=(12, 6))
    for ki, k in enumerate(K_VALUES):
        means = []
        for btl in btl_levels:
            diffs = [p["k_results"][str(k)]["mean_logit_diff"] for p in all_data[btl]]
            means.append(np.mean(diffs))
        ax.plot(range(len(btl_levels)), means, "s-", label=f"k={k}", color=colors[ki], linewidth=2, markersize=8)
    ax.set_xticks(range(len(btl_levels))); ax.set_xticklabels(btl_short, fontsize=11)
    ax.set_xlabel("BTL Level"); ax.set_ylabel("Mean |Δ Logit|")
    ax.set_title("Mean Logit Difference vs BTL Level", fontsize=14, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(plot_dir, "logit_diff_vs_btl.png"), dpi=150); plt.show(); plt.close()

    # ── 3. Effective Rank vs BTL Level (per layer, at k=1) ──
    fig, ax = plt.subplots(figsize=(12, 6))
    layer_colors = plt.cm.tab10(np.linspace(0, 1, len(resolved_layers)))
    for li, layer_idx in enumerate(resolved_layers):
        means, stds = [], []
        for btl in btl_levels:
            ranks = [p["k_results"]["1"]["layers"][str(layer_idx)]["mean_effective_rank"]
                     for p in all_data[btl] if str(layer_idx) in p["k_results"]["1"]["layers"]]
            means.append(np.mean(ranks)); stds.append(np.std(ranks))
        ax.errorbar(range(len(btl_levels)), means, yerr=stds, marker="o", label=f"Layer {layer_idx}",
                    color=layer_colors[li], linewidth=2, capsize=4)
    ax.set_xticks(range(len(btl_levels))); ax.set_xticklabels(btl_short, fontsize=11)
    ax.set_xlabel("BTL Level"); ax.set_ylabel("Mean Effective Rank")
    ax.set_title("Effective Rank vs BTL Level (per layer, k=1)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(plot_dir, "eff_rank_vs_btl.png"), dpi=150); plt.show(); plt.close()

    # ── 4. Energy Retention vs BTL Level (per layer, k=1) ──
    fig, ax = plt.subplots(figsize=(12, 6))
    for li, layer_idx in enumerate(resolved_layers):
        means = []
        for btl in btl_levels:
            energies = [p["k_results"]["1"]["layers"][str(layer_idx)]["mean_energy_retained_pct"]
                        for p in all_data[btl] if str(layer_idx) in p["k_results"]["1"]["layers"]]
            means.append(np.mean(energies))
        ax.plot(range(len(btl_levels)), means, "D-", label=f"Layer {layer_idx}",
                color=layer_colors[li], linewidth=2, markersize=8)
    ax.set_xticks(range(len(btl_levels))); ax.set_xticklabels(btl_short, fontsize=11)
    ax.set_xlabel("BTL Level"); ax.set_ylabel("Energy Retained (%) at k=1")
    ax.set_title("Energy Retention vs BTL Level (k=1)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(plot_dir, "energy_vs_btl.png"), dpi=150); plt.show(); plt.close()

    # ── 5. Spectral Entropy vs BTL Level (per layer, k=1) ──
    fig, ax = plt.subplots(figsize=(12, 6))
    for li, layer_idx in enumerate(resolved_layers):
        means, stds = [], []
        for btl in btl_levels:
            ents = [p["k_results"]["1"]["layers"][str(layer_idx)]["mean_spectral_entropy"]
                    for p in all_data[btl] if str(layer_idx) in p["k_results"]["1"]["layers"]]
            means.append(np.mean(ents)); stds.append(np.std(ents))
        ax.errorbar(range(len(btl_levels)), means, yerr=stds, marker="s", label=f"Layer {layer_idx}",
                    color=layer_colors[li], linewidth=2, capsize=4)
    ax.set_xticks(range(len(btl_levels))); ax.set_xticklabels(btl_short, fontsize=11)
    ax.set_xlabel("BTL Level"); ax.set_ylabel("Spectral Entropy (bits)")
    ax.set_title("Spectral Entropy vs BTL Level (k=1)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(plot_dir, "entropy_vs_btl.png"), dpi=150); plt.show(); plt.close()

    # ── 6. Top-1 Dominance vs BTL Level (per layer, k=1) ──
    fig, ax = plt.subplots(figsize=(12, 6))
    for li, layer_idx in enumerate(resolved_layers):
        means = []
        for btl in btl_levels:
            doms = [p["k_results"]["1"]["layers"][str(layer_idx)]["mean_top1_dominance"]
                    for p in all_data[btl] if str(layer_idx) in p["k_results"]["1"]["layers"]]
            means.append(np.mean(doms))
        ax.plot(range(len(btl_levels)), means, "^-", label=f"Layer {layer_idx}",
                color=layer_colors[li], linewidth=2, markersize=8)
    ax.set_xticks(range(len(btl_levels))); ax.set_xticklabels(btl_short, fontsize=11)
    ax.set_xlabel("BTL Level"); ax.set_ylabel("Top-1 Dominance (σ₁/Σσ)")
    ax.set_title("Top-1 Singular Value Dominance vs BTL Level", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(plot_dir, "dominance_vs_btl.png"), dpi=150); plt.show(); plt.close()

    # ── 7. Heatmap: Effective Rank (BTL × Layer) at k=1 ──
    fig, ax = plt.subplots(figsize=(10, 5))
    matrix = []
    for btl in btl_levels:
        row = []
        for layer_idx in resolved_layers:
            ranks = [p["k_results"]["1"]["layers"][str(layer_idx)]["mean_effective_rank"]
                     for p in all_data[btl] if str(layer_idx) in p["k_results"]["1"]["layers"]]
            row.append(np.mean(ranks))
        matrix.append(row)
    matrix = np.array(matrix)
    im = ax.imshow(matrix, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(resolved_layers))); ax.set_xticklabels([f"L{l}" for l in resolved_layers])
    ax.set_yticks(range(len(btl_levels))); ax.set_yticklabels(btl_short)
    ax.set_xlabel("Layer"); ax.set_ylabel("BTL Level")
    ax.set_title("Mean Effective Rank (BTL × Layer, k=1)", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    for i in range(len(btl_levels)):
        for j in range(len(resolved_layers)):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if matrix[i,j] < np.mean(matrix) else "black")
    plt.tight_layout(); plt.savefig(os.path.join(plot_dir, "heatmap_rank_btl_layer.png"), dpi=150); plt.show(); plt.close()

    # ── 8. Heatmap: KL Divergence (BTL × k) ──
    fig, ax = plt.subplots(figsize=(8, 5))
    matrix = []
    for btl in btl_levels:
        row = []
        for k in K_VALUES:
            kls = [p["k_results"][str(k)]["kl_divergence"] for p in all_data[btl]]
            row.append(np.mean(kls))
        matrix.append(row)
    matrix = np.array(matrix)
    im = ax.imshow(matrix, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(K_VALUES))); ax.set_xticklabels([f"k={k}" for k in K_VALUES])
    ax.set_yticks(range(len(btl_levels))); ax.set_yticklabels(btl_short)
    ax.set_xlabel("top_k"); ax.set_ylabel("BTL Level")
    ax.set_title("KL Divergence (BTL × k)", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    for i in range(len(btl_levels)):
        for j in range(len(K_VALUES)):
            ax.text(j, i, f"{matrix[i,j]:.4f}", ha="center", va="center", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(plot_dir, "heatmap_kl_btl_k.png"), dpi=150); plt.show(); plt.close()

    # ── 9. Sequence Length vs Effective Rank scatter ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for bi, btl in enumerate(btl_levels):
        seq_lens = [p["seq_len"] for p in all_data[btl]]
        ranks = [p["k_results"]["1"]["layers"][str(resolved_layers[-1])]["mean_effective_rank"]
                 for p in all_data[btl] if str(resolved_layers[-1]) in p["k_results"]["1"]["layers"]]
        ax.scatter(seq_lens[:len(ranks)], ranks, label=btl_short[bi], s=60, alpha=0.7)
    ax.set_xlabel("Sequence Length (tokens)"); ax.set_ylabel("Effective Rank (L31, k=1)")
    ax.set_title("Sequence Length vs Effective Rank (Last Layer)", fontsize=14, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(plot_dir, "seqlen_vs_rank.png"), dpi=150); plt.show(); plt.close()

    # ── 10. Text generation match rate per BTL ──
    if RUN_GENERATION:
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(btl_levels))
        width = 0.15
        for ki, k in enumerate(K_VALUES):
            rates = []
            for btl in btl_levels:
                matches = [1 if p["k_results"][str(k)].get("same_as_baseline") else 0 for p in all_data[btl]]
                rates.append(np.mean(matches) * 100)
            ax.bar(x + ki * width, rates, width, label=f"k={k}", color=colors[ki])
        ax.set_xticks(x + width * 2); ax.set_xticklabels(btl_short)
        ax.set_xlabel("BTL Level"); ax.set_ylabel("% Identical to Baseline")
        ax.set_title("Generation Match Rate vs BTL Level", fontsize=14, fontweight="bold")
        ax.legend(); ax.set_ylim(0, 110); ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout(); plt.savefig(os.path.join(plot_dir, "gen_match_vs_btl.png"), dpi=150); plt.show(); plt.close()


def print_summary_table(all_data, resolved_layers):
    """Print final summary tables."""
    btl_levels = list(BTL_PROMPTS.keys())

    print("\n" + "=" * 90)
    print("  📊 AGGREGATE SUMMARY: SVD × BTL")
    print("=" * 90)

    # Table 1: KL and logit diff per BTL per k
    print(f"\n  {'BTL Level':<16} | {'k':>3} | {'Mean KL':>10} | {'Mean Δ':>10} | {'Max Δ':>10} | {'Top1 Match%':>11}")
    print("  " + "-" * 75)
    for btl in btl_levels:
        short = btl.split("_")[1]
        for k in K_VALUES:
            kls = [p["k_results"][str(k)]["kl_divergence"] for p in all_data[btl]]
            mds = [p["k_results"][str(k)]["mean_logit_diff"] for p in all_data[btl]]
            xds = [p["k_results"][str(k)]["max_logit_diff"] for p in all_data[btl]]
            t1s = [1 if p["k_results"][str(k)]["top1_match"] else 0 for p in all_data[btl]]
            print(f"  {short:<16} | {k:>3} | {np.mean(kls):>10.6f} | {np.mean(mds):>10.6f} | "
                  f"{np.mean(xds):>10.4f} | {np.mean(t1s)*100:>10.1f}%")
        print("  " + "-" * 75)

    # Table 2: Per-layer spectral metrics at k=1
    print(f"\n  {'BTL':<12} | {'Layer':>5} | {'Eff Rank':>9} | {'Entropy':>8} | {'Energy%':>8} | {'σ₁ Dom':>7} | {'#Signif':>7}")
    print("  " + "-" * 70)
    for btl in btl_levels:
        short = btl.split("_")[1][:8]
        for layer_idx in resolved_layers:
            ranks = [p["k_results"]["1"]["layers"][str(layer_idx)]["mean_effective_rank"]
                     for p in all_data[btl] if str(layer_idx) in p["k_results"]["1"]["layers"]]
            ents = [p["k_results"]["1"]["layers"][str(layer_idx)]["mean_spectral_entropy"]
                    for p in all_data[btl] if str(layer_idx) in p["k_results"]["1"]["layers"]]
            engs = [p["k_results"]["1"]["layers"][str(layer_idx)]["mean_energy_retained_pct"]
                    for p in all_data[btl] if str(layer_idx) in p["k_results"]["1"]["layers"]]
            doms = [p["k_results"]["1"]["layers"][str(layer_idx)]["mean_top1_dominance"]
                    for p in all_data[btl] if str(layer_idx) in p["k_results"]["1"]["layers"]]
            nsig = [p["k_results"]["1"]["layers"][str(layer_idx)]["mean_num_significant_sv"]
                    for p in all_data[btl] if str(layer_idx) in p["k_results"]["1"]["layers"]]
            if ranks:
                print(f"  {short:<12} | {layer_idx:>5} | {np.mean(ranks):>9.3f} | {np.mean(ents):>8.3f} | "
                      f"{np.mean(engs):>7.1f}% | {np.mean(doms):>7.3f} | {np.mean(nsig):>7.1f}")
        print("  " + "-" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  SVD Attention Intervention × BTL CoT — Phi-2")
print("=" * 70)

# Load model once
print("\n📦 Loading Phi-2 (once)...")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    trust_remote_code=True, attn_implementation="eager",
).to(DEVICE)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

num_layers = len(model.model.layers)
resolved_layers = [i if i >= 0 else num_layers + i for i in TARGET_LAYERS]
print(f"   ✅ Loaded. Layers: {resolved_layers}")

# Check for existing results (resume support)
results_file = os.path.join(OUTPUT_DIR, "data", "all_results.json")
if os.path.exists(results_file):
    print(f"\n📂 Found existing results file, loading...")
    with open(results_file) as f:
        all_data = json.load(f)
    # Count completed
    done = sum(len(v) for v in all_data.values())
    print(f"   {done}/42 prompts already completed. Resuming...")
else:
    all_data = {btl: [] for btl in BTL_PROMPTS}

# Run experiment
total = sum(len(v) for v in BTL_PROMPTS.values())
completed = sum(len(v) for v in all_data.values())
print(f"\n🔬 Running {total - completed} remaining prompts (of {total} total)...")

start_time = time.time()

for btl_level, prompts in BTL_PROMPTS.items():
    already_done = len(all_data[btl_level])
    if already_done >= len(prompts):
        print(f"   ✅ {btl_level}: all {len(prompts)} done, skipping")
        continue

    for idx in range(already_done, len(prompts)):
        prompt = prompts[idx]
        prompt_id = f"{btl_level}_P{idx}"
        elapsed = time.time() - start_time
        prompts_done = sum(len(v) for v in all_data.values())
        rate = elapsed / max(prompts_done - completed + 1, 1)

        print(f"\n{'─'*60}")
        print(f"  [{prompts_done+1}/{total}] {prompt_id}")
        print(f"  \"{prompt[:70]}...\"")
        if prompts_done > completed:
            remaining = (total - prompts_done - 1) * rate
            print(f"  ⏱ {elapsed/60:.1f}min elapsed, ~{remaining/60:.1f}min remaining")

        result = run_single_prompt(model, tokenizer, prompt, resolved_layers)
        result["prompt_id"] = prompt_id
        result["btl_level"] = btl_level
        all_data[btl_level].append(result)

        # Save incrementally
        if SAVE_INCREMENTALLY:
            with open(results_file, "w") as f:
                json.dump(all_data, f)
            # Also save individual
            with open(os.path.join(OUTPUT_DIR, "data", f"{prompt_id}.json"), "w") as f:
                json.dump(result, f)

        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

total_time = time.time() - start_time
print(f"\n✅ All prompts completed in {total_time/60:.1f} minutes")

# Save final
with open(results_file, "w") as f:
    json.dump(all_data, f, indent=2)
print(f"   Saved to {results_file}")

# Generate plots
print("\n🎨 Generating aggregate plots...")
plot_btl_comparison(all_data, resolved_layers)

# Print tables
print_summary_table(all_data, resolved_layers)

print(f"\n✅ Complete! All outputs in: {OUTPUT_DIR}")
