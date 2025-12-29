import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt

# === CONFIG ===
data_path = "data/counting_pca3"
token_mode = "element"      # or "separator"
layer_index = 22            # <--- SELECT LAYER HERE
output_format = "pdf"       # save as PDF
out_name = f"layer_{layer_index:02d}_cosine_with_ref_{token_mode}.{output_format}"

# === FUNCTIONS (unchanged) ===
def load_token_embeddings(layer_path, token_type):
    data = {}
    for file in os.listdir(layer_path):
        if not file.endswith(".pt") or not file.startswith(token_type):
            continue
        m = re.search(r"_(\d+)\.pt$", file)
        if not m:
            continue
        token_idx = int(m.group(1))
        vec = torch.load(os.path.join(layer_path, file), map_location="cpu")
        data[token_idx] = vec.detach().cpu().numpy().flatten()
    return data

def compute_reference_diffs(token_data):
    diffs = []
    ref = token_data.get(0)
    if ref is None:
        return None
    for k in range(1, 9):
        if k not in token_data:
            return None
        diff = token_data[k] - ref
        norm = np.linalg.norm(diff)
        if norm > 0:
            diff /= norm
        diffs.append(diff)
    return np.stack(diffs)

def layer_cosine_similarity(layer_dirs, token_mode):
    exp_diffs = []
    for layer_dir in layer_dirs:
        tokens = load_token_embeddings(layer_dir, token_mode)
        diffs = compute_reference_diffs(tokens)
        if diffs is not None:
            exp_diffs.append(diffs)

    n_exps = len(exp_diffs)
    if n_exps < 2:
        return np.zeros((8, 8))

    sim_mats = []
    for i in range(n_exps):
        for j in range(i + 1, n_exps):
            v1 = exp_diffs[i]
            v2 = exp_diffs[j]
            sim = v1 @ v2.T
            sim_mats.append(sim)

    sim_mats = np.stack(sim_mats)
    return np.mean(sim_mats, axis=0)

# === LOAD EXPERIMENTS ===
experiments = [
    os.path.join(data_path, exp)
    for exp in os.listdir(data_path)
    if os.path.isdir(os.path.join(data_path, exp))
]

if not experiments:
    raise ValueError(f"No experiments found in {data_path}")

sample_exp = experiments[0]
layer_names = sorted([d for d in os.listdir(sample_exp) if d.startswith("layer_")])

if layer_index >= len(layer_names):
    raise IndexError(f"Layer index {layer_index} out of range ({len(layer_names)} layers found).")

selected_layer = layer_names[layer_index]

# === Compute similarity matrix ===
layer_dirs = [
    os.path.join(exp, selected_layer)
    for exp in experiments
    if os.path.exists(os.path.join(exp, selected_layer))
]

mat = layer_cosine_similarity(layer_dirs, token_mode)
vmin = 0.55
# vmin = np.nanmin(mat)
vmax = np.nanmax(mat)

# === PLOT SINGLE LAYER ===
plt.figure(figsize=(6, 5))
im = plt.imshow(mat, vmin=vmin, vmax=vmax, cmap="coolwarm", origin="lower")

plt.title(f"Cosine Similarity of embeddings (layer:22)",
          fontsize=13, fontweight="bold")

plt.xlabel("Item embeddings of Task A", fontsize=14)
plt.ylabel("Item embeddings of Task B", fontsize=14)

# plt.xticks(range(8), [f"{k:02d}" for k in range(1, 9)])
# plt.yticks(range(8), [f"{k:02d}" for k in range(1, 9)])

plt.colorbar(im, label="Cosine Similarity")

plt.tight_layout()
plt.savefig(out_name, dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ Saved single-layer cosine similarity as {out_name}")
     