import pandas as pd
import matplotlib.pyplot as plt

# Load benchmark CSV
df = pd.read_csv("benchmark_results.csv")

# Convert True/False to numeric
df["Correct"] = df["Correct"].astype(int)

# Compute accuracy by difficulty
accuracy = df.groupby("Difficulty")["Correct"].mean() * 100

# Plot
plt.figure()
plt.plot(accuracy.index, accuracy.values)
plt.title("Benchmark Accuracy by Difficulty")
plt.xlabel("Difficulty")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()
