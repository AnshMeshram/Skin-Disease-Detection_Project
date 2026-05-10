import matplotlib.pyplot as plt
import numpy as np

# Data from project results
models = ['Inception-V3', 'ConvNeXt-Tiny', 'EfficientNet-B3', 'Final Ensemble']
accuracies = [85.0, 88.0, 90.2, 91.4]

# Set professional IEEE style
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

x = np.arange(len(models))
width = 0.5

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Create bars with a gradient feel
colors = ['#94A3B8', '#64748B', '#3B82F6', '#1E40AF']
rects = ax.bar(x, accuracies, width, color=colors, alpha=0.9, edgecolor='black', linewidth=1.2)

# Highlight the ensemble
rects[3].set_edgecolor('gold')
rects[3].set_linewidth(2.5)

ax.set_ylabel('Validation Accuracy (%)')
ax.set_title('Impact of Architectural Diversity on Model Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(80, 95)  # Zoom in on the relevant range

# Add horizontal grid lines
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)

# Add value labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=11)

autolabel(rects)

# Add annotation about the ensemble gain
gain = 91.4 - 90.2
ax.annotate(f'+{gain:.1f}% Ensemble Gain', 
            xy=(3, 91.4), xytext=(2, 93),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
            fontsize=11, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#EFF6FF", ec="#1E40AF", alpha=0.9))

fig.tight_layout()
plt.savefig('outputs/plots/ensemble_comparison_ieee.png', bbox_inches='tight')
print("IEEE level Ensemble Comparison plot generated at outputs/plots/ensemble_comparison_ieee.png")
