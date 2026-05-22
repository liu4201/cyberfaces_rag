import matplotlib.pyplot as plt
import numpy as np


#plt.figure(figsize=(20, 6))

species = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
penguin_means = {
    'jina-reranker-v2': (0.41, 0.40, 0.23, 0.23, 0.21, 0.20, 0.18, 0.13, 0.07, 0.07),
    'jina-reranker-v2 + instruct 1': (0.38, 0.14, 0.14, 0.14, 0.23, 0.19, 0.28, 0.12, 0.03, 0.03),
    'jina-reranker-v2 + instruct 2': (0.47, 0.10, 0.17, 0.15, 0.29, 0.19, 0.29, 0.18, 0.05, 0.04),
    'jina-reranker-v2 + instruct 3': (0.24, 0.10, 0.16, 0.14, 0.29, 0.17, 0.30, 0.10, 0.05, 0.04),
    'jina-reranker-v2 + QE with prompt 1': (0.27, 0.10, 0.15, 0.28, 0.34, 0.25, 0.30, 0.19, 0.10, 0.09),
    'jina-reranker-v2 + QE with prompt 1 + instruct 1': (0.47, 0.13, 0.20, 0.22, 0.41, 0.31, 0.46, 0.32, 0.27, 0.23),
    'jina-reranker-v2 + QE with prompt 1 + instruct 2': (0.48, 0.13, 0.21, 0.32, 0.53, 0.37, 0.53, 0.34, 0.20, 0.18),
    'jina-reranker-v2 + QE with prompt 1 + instruct 3': (0.41, 0.14, 0.21, 0.28, 0.46, 0.39, 0.46, 0.20, 0.21, 0.17),
    'jina-reranker-v2 + QE with prompt 2': (0.32, 0.14, 0.18, 0.22, 0.24, 0.20, 0.21, 0.24, 0.15, 0.12),
    'jina-reranker-v2 + QE with prompt 2 + instruct': (0.35, 0.10, 0.16, 0.29, 0.33, 0.21, 0.34, 0.24, 0.17, 0.12),
    'jina-reranker-v2 + QE with prompt 3': (0.54, 0.51, 0.46, 0.30, 0.27, 0.27, 0.25, 0.16, 0.16, 0.15),
    'jina-reranker-v2 + QE with prompt 3 + instruct 1': (0.45, 0.08, 0.25, 0.35, 0.47, 0.33, 0.47, 0.24, 0.07, 0.08),
    'jina-reranker-v2 + QE with prompt 3 + instruct 2': (0.55, 0.10, 0.28, 0.34, 0.47, 0.33, 0.49, 0.26, 0.12, 0.08),
    'jina-reranker-v2 + QE with prompt 3 + instruct 3': (0.22, 0.05, 0.13, 0.15, 0.21, 0.14, 0.13, 0.13, 0.07, 0.07),
    'jina-reranker-v2 + QE with prompt 3 + instruct 4': (0.24, 0.06, 0.10, 0.12, 0.22, 0.11, 0.19, 0.09, 0.06, 0.05),
    'jina-reranker-v3 + QE with prompt 1 + instruct 2': (-0.02, -0.06, -0.12, -0.09, -0.08, -0.07, -0.09, -0.09, -0.12, -0.09),
    'jina-reranker-v3 + QE with prompt 3 + instruct 2': (0.21, -0.04, -0.10, -0.10, -0.06, -0.07, -0.07, -0.13, -0.09, -0.097),
    #'Direct LLMs': (1, 1, 0.9, 0.9, 1, 0.9, 1, 1, 0.8, 0.8),
}

x = np.arange(len(species))  # the label locations
width = 0.05  # the width of the bars
multiplier = 0

cmap = plt.get_cmap('tab20')
colors = [cmap(i) for i in np.linspace(0, 1, len(penguin_means))]

fig, ax = plt.subplots(figsize=(14, 4.5), layout='constrained')

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[multiplier])
    #ax.bar_label(rects, padding=3)
    multiplier += 1

plt.axhline(y=0.30, color='r', linestyle='--', linewidth=1)
plt.axhline(y=-0.1, color=colors[15], linestyle='--', linewidth=1)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_xlabel('ID of the Documents')
ax.set_title('Ranking Scores of 10 Documents Relevant to "Hydrology"')
ax.set_xticks(x + width, species)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(-0.2, 1)

ax.text(7.5, 0.32, 'Relaxed jina-reranker-v2 Threshold: 0.30', color='r', fontweight='bold')
ax.text(4, -0.15, 'Relaxed jina-reranker-v3 Threshold: -0.1', color=colors[15], fontweight='bold')


plt.savefig('smartsearch_plot.png', dpi=200, bbox_inches='tight')
plt.show()
