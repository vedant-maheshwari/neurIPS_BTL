"""
SVD Attention Intervention for Phi-2 — Multi-Layer Sweep (Kaggle Edition)
=========================================================================
Patches layers 15–20 and 31 simultaneously with top_k modes.
Sweeps k=1→5. Collects per-layer spectral metadata.
"""

import os, gc
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
PROMPT = "Explain the concept of entropy in thermodynamics step by step."
MAX_NEW_TOKENS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "./svd_multilayer_outputs"
TARGET_LAYERS = [15, 16, 17, 18, 19, 20, -1]  # layers 15-20 + last
K_VALUES = [1, 2, 3, 4, 5]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SVD TRUNCATION
# ═══════════════════════════════════════════════════════════════════════════════
def svd_truncate_attention(attn_weights, top_k):
    """SVD-truncate attention: keep top_k modes, zero the rest."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# SPECTRAL METADATA EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════
def compute_spectral_metadata(attn_tensor, top_k):
    """
    Given attention tensor (B, H, Q, K), compute per-head spectral metadata.
    Returns dict with arrays of shape (H,).
    """
    attn = attn_tensor[0].numpy()  # (H, Q, K)
    H = attn.shape[0]

    metadata = {
        "singular_values": [],       # list of arrays, one per head
        "effective_rank": np.zeros(H),
        "spectral_entropy": np.zeros(H),
        "energy_total": np.zeros(H),
        "energy_retained": np.zeros(H),
        "energy_retained_pct": np.zeros(H),
        "top1_dominance": np.zeros(H),  # σ1 / sum(σ)
        "spectral_gap": np.zeros(H),    # σ1 - σ2
        "frobenius_norm": np.zeros(H),
        "nuclear_norm": np.zeros(H),
    }

    for h in range(H):
        A = attn[h]  # (Q, K)
        sv = np.linalg.svd(A, compute_uv=False)
        sv = sv[sv > 1e-30]  # filter numerical zeros

        metadata["singular_values"].append(sv)

        # Energy
        energy = sv ** 2
        total_energy = np.sum(energy)
        metadata["energy_total"][h] = total_energy
        metadata["frobenius_norm"][h] = np.sqrt(total_energy)
        metadata["nuclear_norm"][h] = np.sum(sv)

        # Energy retained by top_k
        k = min(top_k, len(sv))
        retained = np.sum(energy[:k])
        metadata["energy_retained"][h] = retained
        metadata["energy_retained_pct"][h] = (retained / total_energy * 100) if total_energy > 0 else 100.0

        # Spectral entropy: -sum(p_i * log(p_i))
        if total_energy > 0:
            p = energy / total_energy
            p = p[p > 0]
            metadata["spectral_entropy"][h] = -np.sum(p * np.log2(p))
        else:
            metadata["spectral_entropy"][h] = 0.0

        # Effective rank: exp(entropy in nats)
        if total_energy > 0:
            p = energy / total_energy
            p = p[p > 0]
            entropy_nats = -np.sum(p * np.log(p))
            metadata["effective_rank"][h] = np.exp(entropy_nats)
        else:
            metadata["effective_rank"][h] = 1.0

        # Top-1 dominance
        metadata["top1_dominance"][h] = sv[0] / np.sum(sv) if len(sv) > 0 else 1.0

        # Spectral gap
        metadata["spectral_gap"][h] = (sv[0] - sv[1]) if len(sv) > 1 else sv[0]

    return metadata


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM ATTENTION FORWARD WITH SVD INJECTION
# ═══════════════════════════════════════════════════════════════════════════════
def svd_eager_attention_forward(
    module, query, key, value, attention_mask, scaling,
    dropout=0.0, top_k=5, capture_dict=None, layer_id=None, **kwargs,
):
    from transformers.models.phi.modeling_phi import repeat_kv
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # SVD truncation
    original_attn_weights = attn_weights.clone()
    attn_weights = svd_truncate_attention(attn_weights, top_k=top_k)

    # Capture (only on prefill — when Q == K, i.e., square attention)
    if capture_dict is not None and layer_id is not None:
        Q_len = original_attn_weights.shape[2]
        K_len = original_attn_weights.shape[3]
        if Q_len == K_len:  # only capture during prefill, not autoregressive steps
            capture_dict[layer_id] = {
                "original_attn": original_attn_weights.detach().cpu().float(),
                "truncated_attn": attn_weights.detach().cpu().float(),
            }

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-LAYER PATCHER
# ═══════════════════════════════════════════════════════════════════════════════
class MultiLayerSVDPatcher:
    """Patches multiple layers simultaneously with the same top_k."""

    def __init__(self, model, top_k=5, layer_indices=None):
        self.model = model
        self.top_k = top_k
        self.layer_indices = layer_indices or [-1]
        self.capture = {}  # {layer_idx: {"original_attn": ..., "truncated_attn": ...}}
        self._originals = {}  # {layer_idx: original_forward}

    def __enter__(self):
        num_layers = len(self.model.model.layers)
        # Resolve negative indices
        resolved = [idx if idx >= 0 else num_layers + idx for idx in self.layer_indices]

        for layer_idx in resolved:
            target_attn = self.model.model.layers[layer_idx].self_attn
            self._originals[layer_idx] = target_attn.forward

            top_k = self.top_k
            capture = self.capture
            attn_mod = target_attn
            lid = layer_idx

            def make_patched_forward(attn_mod, lid):
                def patched_forward(hidden_states, position_embeddings, attention_mask=None,
                                    past_key_values=None, **kwargs):
                    from transformers.models.phi.modeling_phi import apply_rotary_pos_emb
                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, attn_mod.head_dim)
                    q = attn_mod.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    k = attn_mod.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    v = attn_mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    if getattr(attn_mod, "qk_layernorm", False):
                        q, k = attn_mod.q_layernorm(q), attn_mod.k_layernorm(k)
                    cos, sin = position_embeddings
                    nd = attn_mod.rotary_ndims
                    qr, qp = q[..., :nd], q[..., nd:]
                    kr, kp = k[..., :nd], k[..., nd:]
                    qr, kr = apply_rotary_pos_emb(qr, kr, cos, sin)
                    q = torch.cat((qr, qp), dim=-1)
                    k = torch.cat((kr, kp), dim=-1)
                    if past_key_values is not None:
                        k, v = past_key_values.update(k, v, attn_mod.layer_idx)
                    out, w = svd_eager_attention_forward(
                        attn_mod, q, k, v, attention_mask,
                        dropout=0.0, scaling=attn_mod.scaling,
                        top_k=top_k, capture_dict=capture, layer_id=lid,
                    )
                    out = out.reshape(*input_shape, -1).contiguous()
                    dense = getattr(attn_mod, "dense", getattr(attn_mod, "o_proj", None))
                    out = dense(out)
                    return out, w
                return patched_forward

            target_attn.forward = make_patched_forward(attn_mod, lid)

        return self

    def __exit__(self, *args):
        for layer_idx, orig_fwd in self._originals.items():
            self.model.model.layers[layer_idx].self_attn.forward = orig_fwd
        self._originals.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sv_spectrum_grid(all_metadata, k, layers, save_path):
    """SV spectrum for all target layers at a given k."""
    n_layers = len(layers)
    cols = min(4, n_layers)
    rows = (n_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)

    for i, layer_idx in enumerate(layers):
        ax = axes[i // cols][i % cols]
        meta = all_metadata[layer_idx]
        for h, sv in enumerate(meta["singular_values"]):
            ax.plot(sv, alpha=0.5, linewidth=0.8)
        ax.set_yscale("log")
        ax.set_title(f"Layer {layer_idx}", fontsize=11, fontweight="bold")
        ax.set_xlabel("SV Index"); ax.set_ylabel("Magnitude")
        ax.grid(True, alpha=0.3)

    # Hide unused
    for i in range(n_layers, rows*cols):
        axes[i // cols][i % cols].set_visible(False)

    fig.suptitle(f"Singular Value Spectra (top_k={k})", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.show(); plt.close()


def plot_energy_retention_grid(all_metadata, k, layers, save_path):
    """Energy retention bar chart for all layers at a given k."""
    n_layers = len(layers)
    cols = min(4, n_layers)
    rows = (n_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)

    for i, layer_idx in enumerate(layers):
        ax = axes[i // cols][i % cols]
        meta = all_metadata[layer_idx]
        er = meta["energy_retained_pct"]
        H = len(er)
        colors = plt.cm.viridis(er / 100)
        ax.bar(range(H), er, color=colors, edgecolor="black", linewidth=0.3)
        ax.axhline(y=100, color="red", linestyle="--", alpha=0.4)
        ax.set_title(f"Layer {layer_idx} (mean={np.mean(er):.1f}%)", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 105); ax.set_xlabel("Head"); ax.set_ylabel("Energy %")
        ax.grid(True, alpha=0.3, axis="y")

    for i in range(n_layers, rows*cols):
        axes[i // cols][i % cols].set_visible(False)

    fig.suptitle(f"Energy Retained per Head (top_k={k})", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.show(); plt.close()


def plot_metadata_summary(all_metadata, k, layers, save_path):
    """Summary heatmaps: effective rank, entropy, dominance across layers × heads."""
    metrics = ["effective_rank", "spectral_entropy", "top1_dominance", "spectral_gap"]
    titles = ["Effective Rank", "Spectral Entropy (bits)", "Top-1 Dominance (σ₁/Σσ)", "Spectral Gap (σ₁ − σ₂)"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2][idx % 2]
        # Build matrix: layers × heads
        data = []
        for layer_idx in layers:
            data.append(all_metadata[layer_idx][metric])
        data = np.array(data)  # (n_layers, n_heads)

        im = ax.imshow(data, cmap="magma", aspect="auto")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"L{l}" for l in layers])
        ax.set_xlabel("Head Index"); ax.set_ylabel("Layer")
        ax.set_title(title, fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Spectral Metadata Heatmaps (top_k={k})", fontsize=15, fontweight="bold")
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.show(); plt.close()


def plot_attn_comparison_for_layer(capture_layer, head_idx, tokens, layer_idx, save_path):
    """Original vs truncated attention for one layer, one head."""
    orig = capture_layer["original_attn"][0, head_idx].numpy()
    trunc = capture_layer["truncated_attn"][0, head_idx].numpy()
    diff = orig - trunc
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = [t[:10] for t in tokens]
    n = len(labels)
    for ax, data, title in zip(axes, [orig, trunc, diff],
        ["Original", "SVD-Truncated", "Difference"]):
        im = ax.imshow(data, cmap="viridis", aspect="auto")
        ax.set_title(title, fontsize=11, fontweight="bold")
        if n <= 30:
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_yticklabels(labels, fontsize=6)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"Layer {layer_idx}, Head {head_idx}", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(save_path, dpi=120, bbox_inches="tight"); plt.show(); plt.close()


def plot_k_sweep_summary(all_results, baseline_text, save_path):
    """Final comparison plot: KL div, mean logit diff vs k."""
    ks = [r["k"] for r in all_results]
    kls = [r["kl"] for r in all_results]
    mean_diffs = [r["mean_diff"] for r in all_results]
    max_diffs = [r["max_diff"] for r in all_results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(ks, kls, "o-", color="#e74c3c", linewidth=2, markersize=8)
    axes[0].set_xlabel("top_k"); axes[0].set_ylabel("KL Divergence")
    axes[0].set_title("KL Divergence vs top_k", fontweight="bold")
    axes[0].grid(True, alpha=0.3); axes[0].set_xticks(ks)

    axes[1].plot(ks, mean_diffs, "s-", color="#3498db", linewidth=2, markersize=8)
    axes[1].set_xlabel("top_k"); axes[1].set_ylabel("Mean |Δ Logit|")
    axes[1].set_title("Mean Logit Difference vs top_k", fontweight="bold")
    axes[1].grid(True, alpha=0.3); axes[1].set_xticks(ks)

    axes[2].plot(ks, max_diffs, "D-", color="#2ecc71", linewidth=2, markersize=8)
    axes[2].set_xlabel("top_k"); axes[2].set_ylabel("Max |Δ Logit|")
    axes[2].set_title("Max Logit Difference vs top_k", fontweight="bold")
    axes[2].grid(True, alpha=0.3); axes[2].set_xticks(ks)

    fig.suptitle("SVD Truncation Impact Across Modes", fontsize=15, fontweight="bold")
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.show(); plt.close()


def plot_layer_energy_vs_k(all_k_metadata, layers, save_path):
    """Line plot: mean energy retained per layer across k values."""
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.tab10

    for i, layer_idx in enumerate(layers):
        means = []
        for k in K_VALUES:
            meta = all_k_metadata[k][layer_idx]
            means.append(np.mean(meta["energy_retained_pct"]))
        ax.plot(K_VALUES, means, "o-", color=cmap(i), linewidth=2,
                markersize=8, label=f"Layer {layer_idx}")

    ax.set_xlabel("top_k", fontsize=12); ax.set_ylabel("Mean Energy Retained (%)", fontsize=12)
    ax.set_title("Mean Energy Retained vs top_k (per layer)", fontsize=14, fontweight="bold")
    ax.set_xticks(K_VALUES); ax.set_ylim(0, 105)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.show(); plt.close()


def plot_effective_rank_vs_k(all_k_metadata, layers, save_path):
    """Line plot: mean effective rank per layer across k values."""
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.tab10

    for i, layer_idx in enumerate(layers):
        means = []
        for k in K_VALUES:
            meta = all_k_metadata[k][layer_idx]
            means.append(np.mean(meta["effective_rank"]))
        ax.plot(K_VALUES, means, "s-", color=cmap(i), linewidth=2,
                markersize=8, label=f"Layer {layer_idx}")

    ax.set_xlabel("top_k", fontsize=12); ax.set_ylabel("Mean Effective Rank", fontsize=12)
    ax.set_title("Mean Effective Rank vs top_k (per layer)", fontsize=14, fontweight="bold")
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.show(); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  SVD Attention Intervention — Phi-2 Multi-Layer Sweep")
print("=" * 70)

# ── Load model once ──
print("\n📦 Loading Phi-2 (once)...")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    trust_remote_code=True,
    attn_implementation="eager",
).to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

num_layers = len(model.model.layers)
resolved_layers = [idx if idx >= 0 else num_layers + idx for idx in TARGET_LAYERS]
print(f"   ✅ Loaded. {num_layers} layers total.")
print(f"   Target layers: {resolved_layers}\n")

# ── Tokenize ──
inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
input_ids = inputs["input_ids"]
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
seq_len = input_ids.shape[1]
print(f"📝 Prompt: \"{PROMPT}\"")
print(f"   Tokenized: {seq_len} tokens\n")

# ── Baseline ──
print("🔵 BASELINE (no intervention)...")
with torch.no_grad():
    baseline_out = model(**inputs)
baseline_logits = baseline_out.logits
baseline_next = baseline_logits[0, -1, :]
baseline_probs = torch.softmax(baseline_next, dim=0)
baseline_top5 = torch.topk(baseline_next, 5)

print("   Top-5 predictions:")
for i, (idx, logit) in enumerate(zip(baseline_top5.indices, baseline_top5.values)):
    print(f"     {i+1}. '{tokenizer.decode(idx)}' (logit={logit.item():.4f})")

print("\n   Generating baseline text...")
with torch.no_grad():
    baseline_gen = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS,
                                  do_sample=False, pad_token_id=tokenizer.eos_token_id)
baseline_text = tokenizer.decode(baseline_gen[0], skip_special_tokens=True)
print(f"   📄 {baseline_text}\n")

# ── Storage ──
all_results = []           # per-k summary
all_k_metadata = {}        # {k: {layer_idx: metadata_dict}}
all_k_captures = {}        # {k: {layer_idx: {"original_attn": ..., "truncated_attn": ...}}}

# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP k = 1 → 5
# ═══════════════════════════════════════════════════════════════════════════════
for top_k in K_VALUES:
    print("\n" + "=" * 70)
    print(f"  🟡 SVD Intervention: top_k = {top_k} | Layers: {resolved_layers}")
    print("=" * 70)

    # ── Forward pass with SVD ──
    with torch.no_grad():
        with MultiLayerSVDPatcher(model, top_k=top_k, layer_indices=TARGET_LAYERS) as patcher:
            mod_out = model(**inputs)
            capture = dict(patcher.capture)  # {layer_idx: {orig, trunc}}

    mod_logits = mod_out.logits
    mod_next = mod_logits[0, -1, :]
    mod_top5 = torch.topk(mod_next, 5)

    print(f"\n   Modified top-5 (k={top_k}):")
    for i, (idx, logit) in enumerate(zip(mod_top5.indices, mod_top5.values)):
        print(f"     {i+1}. '{tokenizer.decode(idx)}' (logit={logit.item():.4f})")

    # Metrics
    logit_diff = (baseline_logits - mod_logits).abs()
    mod_probs = torch.softmax(mod_next, dim=0)
    kl = torch.sum(baseline_probs * (torch.log(baseline_probs + 1e-10) - torch.log(mod_probs + 1e-10))).item()
    mean_diff = logit_diff.mean().item()
    max_diff = logit_diff.max().item()

    print(f"\n   📊 Mean logit diff: {mean_diff:.6f}")
    print(f"   📊 Max logit diff:  {max_diff:.6f}")
    print(f"   📊 KL divergence:   {kl:.6f}")

    # ── Generation ──
    with torch.no_grad():
        with MultiLayerSVDPatcher(model, top_k=top_k, layer_indices=TARGET_LAYERS):
            mod_gen = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=False, pad_token_id=tokenizer.eos_token_id)
    mod_text = tokenizer.decode(mod_gen[0], skip_special_tokens=True)
    same = mod_text == baseline_text
    print(f"\n   📄 Generated: {mod_text}")
    print(f"   {'✅ Same as baseline' if same else '❌ DIFFERENT from baseline'}")

    all_results.append({
        "k": top_k, "mean_diff": mean_diff, "max_diff": max_diff,
        "kl": kl, "text": mod_text, "same": same,
    })

    # ── Compute spectral metadata per layer ──
    k_metadata = {}
    print(f"\n   🔬 Per-layer spectral metadata (k={top_k}):")
    print(f"   {'Layer':>6} | {'Eff Rank':>9} | {'Entropy':>8} | {'Energy%':>8} | {'σ₁ Dom':>7} | {'Spec Gap':>9}")
    print(f"   " + "-" * 62)

    for layer_idx in resolved_layers:
        if layer_idx in capture:
            meta = compute_spectral_metadata(capture[layer_idx]["original_attn"], top_k)
            k_metadata[layer_idx] = meta
            print(f"   {layer_idx:>6} | {np.mean(meta['effective_rank']):>9.3f} | "
                  f"{np.mean(meta['spectral_entropy']):>8.3f} | "
                  f"{np.mean(meta['energy_retained_pct']):>7.1f}% | "
                  f"{np.mean(meta['top1_dominance']):>7.3f} | "
                  f"{np.mean(meta['spectral_gap']):>9.4f}")

    all_k_metadata[top_k] = k_metadata
    all_k_captures[top_k] = capture

    # ── Per-k plots ──
    k_dir = os.path.join(OUTPUT_DIR, f"k{top_k}")
    os.makedirs(k_dir, exist_ok=True)

    if k_metadata:
        plot_sv_spectrum_grid(k_metadata, top_k, resolved_layers, os.path.join(k_dir, "sv_spectra.png"))
        plot_energy_retention_grid(k_metadata, top_k, resolved_layers, os.path.join(k_dir, "energy_retention.png"))
        plot_metadata_summary(k_metadata, top_k, resolved_layers, os.path.join(k_dir, "metadata_heatmaps.png"))

        # Attention comparison for head 0 of first and last target layer
        for layer_idx in [resolved_layers[0], resolved_layers[-1]]:
            if layer_idx in capture:
                plot_attn_comparison_for_layer(
                    capture[layer_idx], 0, tokens, layer_idx,
                    os.path.join(k_dir, f"attn_L{layer_idx}_H0.png"))

    # ── Cleanup per-k intermediates ──
    del mod_out, mod_logits, mod_next, mod_probs, logit_diff, mod_gen
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-k VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  🎨 Generating cross-k comparison plots...")
print("=" * 70)

plot_k_sweep_summary(all_results, baseline_text, os.path.join(OUTPUT_DIR, "k_sweep_summary.png"))
plot_layer_energy_vs_k(all_k_metadata, resolved_layers, os.path.join(OUTPUT_DIR, "energy_vs_k_per_layer.png"))
plot_effective_rank_vs_k(all_k_metadata, resolved_layers, os.path.join(OUTPUT_DIR, "eff_rank_vs_k_per_layer.png"))


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  📊 FINAL SUMMARY: Multi-Layer SVD Truncation Sweep")
print("=" * 70)
print(f"\n  Prompt: \"{PROMPT[:60]}...\"")
print(f"  Layers patched: {resolved_layers}")
print(f"  Sequence length: {seq_len} tokens\n")

print(f"  {'k':>3} | {'Mean Δ':>10} | {'Max Δ':>10} | {'KL Div':>10} | {'Same?':>5} | First 60 chars of generated text")
print("  " + "-" * 100)
for r in all_results:
    same_str = "✅" if r["same"] else "❌"
    text_preview = r["text"][:60].replace("\n", "↵")
    print(f"  {r['k']:>3} | {r['mean_diff']:>10.6f} | {r['max_diff']:>10.6f} | {r['kl']:>10.6f} | {same_str:>5} | {text_preview}")

# Per-layer metadata summary table
print(f"\n  📋 Per-Layer Mean Metrics (across all heads):")
print(f"  {'k':>3} | {'Layer':>5} | {'Eff Rank':>9} | {'Entropy':>8} | {'Energy%':>8} | {'σ₁ Dom':>7} | {'Gap':>7}")
print("  " + "-" * 65)
for k in K_VALUES:
    for layer_idx in resolved_layers:
        if layer_idx in all_k_metadata.get(k, {}):
            m = all_k_metadata[k][layer_idx]
            print(f"  {k:>3} | {layer_idx:>5} | {np.mean(m['effective_rank']):>9.3f} | "
                  f"{np.mean(m['spectral_entropy']):>8.3f} | "
                  f"{np.mean(m['energy_retained_pct']):>7.1f}% | "
                  f"{np.mean(m['top1_dominance']):>7.3f} | "
                  f"{np.mean(m['spectral_gap']):>7.4f}")
    print("  " + "-" * 65)

# Cleanup large captures
del all_k_captures
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()

print("\n✅ Complete! All outputs saved to:", OUTPUT_DIR)
