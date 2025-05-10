import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

np.random.seed(42)
chunk_sizes = np.random.normal(loc=490, scale=10, size=10000)
chunk_sizes = np.clip(chunk_sizes, 250, 500)

# Plot
plt.figure(figsize=(10, 6))
sns.histplot(chunk_sizes, bins=20, kde=True, color="orange", edgecolor="black", alpha=0.4)
plt.title("Estimated Distribution of Chunk Sizes", fontsize=14)
plt.xlabel("Number of Characters per Chunk", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(False)
plt.tight_layout()
plt.show()