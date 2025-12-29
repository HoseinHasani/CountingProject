import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.cm import get_cmap

# === CONFIGURATION ===
data_path = "data/counting_pca2"
layer_index = 22                  # <-- select layer number here
mean_over_dataset = False
label_mode = "number"             # "fruit" or "number"
output_format = "pdf"             # pdf, png, etc.

# === LOAD EXPERIMENTS ===
experiments = [
    os.path.join(data_path, d)
    for d in os.listdir(data_path)
    if os.path.isdir(os.path.join(data_path, d))
]

if not experiments:
    raise ValueError(f"No experiments found in {data_path}")

sample_exp = experiments[0]
layer_files = sorted(
    [f for f in os.listdir(sample_exp) if f.startswith("layer_") and f.endswith(".pt")]
)
layer_names = [f[:-3] for f in layer_files]

if layer_index >= len(layer_names):
    raise IndexError(f"Layer index {layer_index} out of range ({len(layer_names)} layers found).")

selected_layer = layer_names[layer_index]

# === LABEL FUNCTION ===
def get_label(exp_path):
    base = os.path.basename(exp_path)
    parts = base.split("_")
    if label_mode == "fruit":
        return parts[0]
    elif label_mode == "number":
        return parts[-1]
    else:
        raise ValueError(f"Invalid label_mode: {label_mode}")

labels_all = [get_label(e) for e in experiments]

# === LOAD EMBEDDINGS ===
def load_layer_embeddings(experiments, layer_name, mean_over_dataset):
    layer_vecs = []
    for exp in experiments:
        layer_file = os.path.join(exp, layer_name + ".pt")
        if not os.path.exists(layer_file):
            continue
        tensor = torch.load(layer_file, map_location="cpu")
        if tensor.dtype in (torch.bfloat16, torch.float16):
            tensor = tensor.to(torch.float32)
        tensor = tensor.detach().cpu().numpy().reshape(-1)
        layer_vecs.append(tensor)
    if not layer_vecs:
        return None
    if mean_over_dataset:
        return np.mean(np.stack(layer_vecs), axis=0, keepdims=True)
    else:
        return np.stack(layer_vecs)

X = load_layer_embeddings(experiments, selected_layer, mean_over_dataset)
if X is None:
    raise ValueError(f"No data found for layer {selected_layer}")

# === PCA AND PLOT ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

unique_labels = sorted(set(labels_all))
cmap = get_cmap("tab10")
color_map = {label: cmap(i / max(1, len(unique_labels) - 1))
             for i, label in enumerate(unique_labels)}

plt.figure(figsize=(6, 5))
for (x, y), lbl in zip(X_pca, labels_all):
    plt.scatter(x, y, color=color_map[lbl], edgecolor="black", s=80, alpha=0.9)

plt.title(f"PCA of Layer Embeddings (layer: 22)", fontsize=16, fontweight="bold")
plt.xlabel("Principal Component 1", fontsize=15)
plt.ylabel("Principal Component 2", fontsize=15)

handles = [
    plt.Line2D([], [], color=cmap(i / max(1, len(unique_labels) - 1)),
               marker='o', linestyle='', label=str(lbl))
    for i, lbl in enumerate(unique_labels)
]
plt.legend(handles=handles, title="Labels", frameon=False, fontsize=14)

plt.tight_layout()

out_name = f"layer_{layer_index:02d}_pca_{label_mode}.{output_format}"
plt.savefig(out_name, dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ Saved PCA visualization for {selected_layer} as {out_name}")
