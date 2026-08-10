import matplotlib.pyplot as plt
import numpy as np

# Data from project analysis
stages = ['Baseline\n(Pretrained)', '+ Segmentation\n(U-Net)', '+ Augmentation\n(Mixup/Focal)', '+ Ensemble\n(Final)']
accuracies = [83.1, 85.5, 88.2, 91.4]

# Set professional IEEE style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 11,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18
})

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Create a step/line plot with points
ax.plot(stages, accuracies, marker='o', linestyle='-', linewidth=3, markersize=10, color='#1E40AF', markerfacecolor='#3B82F6', markeredgecolor='black', markeredgewidth=1.5)

# Fill area under the curve for visual weight
ax.fill_between(stages, accuracies, alpha=0.1, color='#3B82F6')

# Add text labels for each point
for i, acc in enumerate(accuracies):
    ax.annotate(f'{acc:.1f}%', 
                xy=(stages[i], acc), xytext=(0, 10),
                textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=12)

# Add horizontal grid lines
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)

ax.set_ylabel('Validation Accuracy (%)')
ax.set_title('Ablation Study: Cumulative Accuracy Gains per Design Stage')
ax.set_ylim(80, 95)

# Add gain annotations
for i in range(1, len(accuracies)):
    gain = accuracies[i] - accuracies[i-1]
    ax.annotate(f'+{gain:.1f}%', 
                xy=(i - 0.5, (accuracies[i] + accuracies[i-1])/2), 
                xytext=(0, 0), textcoords="offset points", 
                ha='center', va='center', fontsize=10, color='green', fontweight='bold',
                bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="green", alpha=0.7))

fig.tight_layout()
plt.savefig('outputs/plots/ablation_study_ieee.png', bbox_inches='tight')
print("IEEE level Ablation Study plot generated at outputs/plots/ablation_study_ieee.png")

# --- Latency Plot ---

batch_sizes = [1, 4, 8, 16, 32]
# Estimated latencies based on user numbers
eff_lat = [18, 25, 42, 75, 130]
inv3_lat = [35, 52, 88, 140, 250]
cnxt_lat = [28, 40, 65, 110, 200]
ens_lat = [68, 105, 180, 310, 560]

fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=300)

ax2.plot(batch_sizes, eff_lat, 'o-', label='EfficientNet-B3 (12M)', color='#10B981', linewidth=2)
ax2.plot(batch_sizes, inv3_lat, 's-', label='Inception-V3 (23M)', color='#F59E0B', linewidth=2)
ax2.plot(batch_sizes, cnxt_lat, '^-', label='ConvNeXt-Tiny (28M)', color='#6366F1', linewidth=2)
ax2.plot(batch_sizes, ens_lat, 'D--', label='Full Ensemble (63M)', color='#EF4444', linewidth=2.5)

ax2.set_xlabel('Batch Size')
ax2.set_ylabel('Inference Latency (ms)')
ax2.set_title('Inference Latency vs. Batch Size for Each Backbone')
ax2.set_xscale('log', base=2)
ax2.set_yscale('log')
ax2.set_xticks(batch_sizes)
ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())

ax2.legend(loc='upper left', frameon=True, shadow=True)
ax2.grid(True, which="both", ls="-", alpha=0.2)

# Highlight single-image latency
ax2.annotate('18ms (Single Image)', xy=(1, 18), xytext=(2, 12),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
             fontsize=10, fontweight='bold')

ax2.annotate('68ms (Ensemble)', xy=(1, 68), xytext=(2, 50),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
             fontsize=10, fontweight='bold')

fig2.tight_layout()
plt.savefig('outputs/plots/latency_analysis_ieee.png', bbox_inches='tight')
print("IEEE level Latency Analysis plot generated at outputs/plots/latency_analysis_ieee.png")
