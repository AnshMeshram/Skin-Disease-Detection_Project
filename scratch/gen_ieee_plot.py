import matplotlib.pyplot as plt
import numpy as np

# Data from previous checkpoint inspection
folds = ['Fold 0', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4']
balanced_acc = [94.04, 78.32, 92.86, 91.66, 95.48]
val_acc = [90.44, 72.74, 83.32, 82.89, 90.32]

# Set professional style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18
})

x = np.arange(len(folds))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

rects1 = ax.bar(x - width/2, balanced_acc, width, label='Balanced Accuracy', color='#2563EB', alpha=0.8, edgecolor='black', linewidth=1)
rects2 = ax.bar(x + width/2, val_acc, width, label='Validation Accuracy', color='#10B981', alpha=0.8, edgecolor='black', linewidth=1)

# Highlight Fold 4
rects1[4].set_color('#1E40AF')
rects1[4].set_edgecolor('gold')
rects1[4].set_linewidth(2)

ax.set_ylabel('Score (%)')
ax.set_title('EfficientNet-B3 K-Fold Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(folds)
ax.set_ylim(60, 105)
ax.legend(loc='lower right', frameon=True, shadow=True)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold' if height > 95 else 'normal')

autolabel(rects1)
autolabel(rects2)

# Add a note about Fold 4
ax.annotate('Best Performing Fold\n(Production Deployment)', 
            xy=(4 - width/2, 95.48), xytext=(2.5, 98),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

fig.tight_layout()
plt.savefig('outputs/plots/kfold_performance_ieee.png', bbox_inches='tight')
print("IEEE level K-Fold performance plot generated at outputs/plots/kfold_performance_ieee.png")
