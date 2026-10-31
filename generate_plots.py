
import os
import matplotlib.pyplot as plt
import numpy as np

# Create plots folder
os.makedirs("plots", exist_ok=True)

# Consistent chart settings
plt.rcParams.update({
    "figure.figsize": (9, 5),
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "savefig.dpi": 180
})


# ==================================================
# 1. CONFUSION MATRIX
# ==================================================

# Final Political-term Ablation SVM
# Rows = Actual labels; Columns = Predicted labels
confusion_matrix = np.array([
    [6545, 56],
    [812, 15682]
])

fig, ax = plt.subplots(figsize=(7, 5))

image = ax.imshow(confusion_matrix, cmap="Blues")

ax.set_title("Final Model - Confusion Matrix", pad=15)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")

ax.set_xticks([0, 1])
ax.set_xticklabels(["Fake News", "Real News"])
ax.set_yticks([0, 1])
ax.set_yticklabels(["Fake News", "Real News"])

for i in range(2):
    for j in range(2):
        value = confusion_matrix[i, j]
        ax.text(
            j, i, f"{value:,}",
            ha="center",
            va="center",
            fontsize=14,
            color="white" if value > confusion_matrix.max() / 2 else "black"
        )

fig.colorbar(image, ax=ax, label="Number of Articles")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png", bbox_inches="tight")
plt.close()


# ==================================================
# 2. MODEL COMPARISON
# ==================================================

models = [
    "Logistic Regression",
    "Multinomial NB",
    "Linear SVM"
]

accuracy = [95.46, 90.40, 96.24]
balanced_accuracy = [96.26, 92.19, 97.11]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))

bars1 = ax.bar(
    x - width / 2,
    accuracy,
    width,
    label="Accuracy"
)

bars2 = ax.bar(
    x + width / 2,
    balanced_accuracy,
    width,
    label="Balanced Accuracy"
)

ax.set_title("Comparison of Machine Learning Models", pad=15)
ax.set_ylabel("Score (%)")
ax.set_xlabel("Classification Algorithm")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(85, 100)
ax.legend()
ax.grid(axis="y", alpha=0.25)

ax.bar_label(bars1, fmt="%.2f%%", padding=3, fontsize=9)
ax.bar_label(bars2, fmt="%.2f%%", padding=3, fontsize=9)

plt.tight_layout()
plt.savefig("plots/model_comparison.png", bbox_inches="tight")
plt.close()


# ==================================================
# 3. DATASET DISTRIBUTION
# ==================================================

# Cleaned dataset after removing empty and duplicate article bodies
labels = ["Fake News", "Real News"]
counts = [17453, 21191]

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    labels,
    counts,
    width=0.55
)

ax.set_title("Distribution of Cleaned News Dataset", pad=15)
ax.set_ylabel("Number of Articles")
ax.grid(axis="y", alpha=0.25)

total = sum(counts)

for bar, count in zip(bars, counts):
    percentage = count / total * 100

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{count:,}\n({percentage:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=11
    )

plt.tight_layout()
plt.savefig("plots/dataset_distribution.png", bbox_inches="tight")
plt.close()


# ==================================================
# 4. PROJECT WORKFLOW DIAGRAM
# ==================================================

fig, ax = plt.subplots(figsize=(12, 3.5))

ax.set_xlim(0, 12)
ax.set_ylim(0, 4)
ax.axis("off")

steps = [
    "News Article",
    "Text\nPreprocessing",
    "TF-IDF\nVectorization",
    "Linear SVM\nClassifier",
    "Prediction"
]

positions = [0.2, 2.6, 5.0, 7.4, 9.8]

for i, (step, x_pos) in enumerate(zip(steps, positions)):

    ax.text(
        x_pos + 0.9,
        2,
        step,
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="white",
            edgecolor="steelblue",
            linewidth=2
        )
    )

    if i < len(steps) - 1:
        ax.annotate(
            "",
            xy=(x_pos + 2.25, 2),
            xytext=(x_pos + 1.75, 2),
            arrowprops=dict(
                arrowstyle="->",
                lw=1.8
            )
        )

ax.set_title("Fake News Detection - Project Workflow", fontsize=16, pad=15)

plt.tight_layout()
plt.savefig("plots/project_workflow.png", bbox_inches="tight")
plt.close()


print("All four project images generated successfully!")
print("Images saved inside the plots folder.")
