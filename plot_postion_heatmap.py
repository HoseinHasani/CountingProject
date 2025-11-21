import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Configuration ===
data_type = 'mono'
# data_type = 'poly'
item_type = 'elements'
# item_type = 'separators'

file_path = f"per_item_data/{data_type}_{item_type}.csv"  # your CSV file
# title = f"Per-Item Latent Count — {data_type.capitalize()}typic — {item_type.capitalize()}"
title = f"Per-Item Latent Count — {data_type.capitalize()}typic"
output_path = f"position_latent_count_{data_type}typic_{item_type}.pdf"


N_item = 10 if item_type == 'elements' else 9

# === Load properly as comma-separated ===
df = pd.read_csv(file_path, sep=",")
df.columns = df.columns.str.strip()  # clean header names

# === Sort and extract ===
df = df.sort_values("num")
data = df[[f"p{i}" for i in range(1, 10)]].to_numpy().T

# === Plot ===
plt.figure(figsize=(6, 5))
sns.set(style="whitegrid", font_scale=1.1)

ax = sns.heatmap(
    data,
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    linewidths=0.8,
    cbar_kws={"label": "Predicted Probability"},
    xticklabels=[f"{i}" for i in range(1, N_item)],
    yticklabels=[f"{i}" for i in range(1, 10)],
    # annot_kws={"size": 8}
)

ax.set_title(title, fontsize=13, pad=14, fontweight='bold')
ax.set_xlabel("Item Position in Sequence", fontsize=13)
ax.set_ylabel("Decoded Number Probability", fontsize=13)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()
