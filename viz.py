import numpy as np
import matplotlib.pyplot as plt

# Load the temperature data from the CSV file
try:
    data = np.loadtxt("heat_output.csv", delimiter=",")
except ValueError as e:
    print("Error loading CSV file:", e)
    with open("heat_output.csv", "r") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if len(line.split(",")) != 500:
                print(f"Row {idx} has an incorrect number of columns.")
    raise

# Create the heatmap
plt.figure(figsize=(10, 8))
plt.imshow(data, cmap="hot", interpolation="nearest")
plt.colorbar(label="Temperature")
plt.title("2D Heat Equation Temperature Distribution")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# Save the visualization as an image (optional)
plt.savefig("heat_distribution.png")

# Display the heatmap
plt.show()