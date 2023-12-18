import numpy as np
from scipy.stats import binned_statistic_2d

# Generate sample 2D data (replace this with your own data)
np.random.seed(42)
x = np.random.uniform(0, 100, size=1000)
y = np.random.uniform(0, 100, size=1000)

# Define the number of bins in x and y directions
bins = 20

# Create the 2D histogram
bin_counts, x_edge, y_edge, bin_number = binned_statistic_2d(x, y, None, 'count', bins=bins)

# Combine adjacent bins until each bin has at least 10 particles
min_particles = 10
while np.min(bin_counts) < min_particles:
    min_count_idx = np.unravel_index(np.argmin(bin_counts), bin_counts.shape)
    neighbors = [
        (min_count_idx[0] + i, min_count_idx[1] + j)
        for i in [-1, 0, 1]
        for j in [-1, 0, 1]
        if 0 <= min_count_idx[0] + i < bins and 0 <= min_count_idx[1] + j < bins
    ]
    for neighbor in neighbors:
        bin_counts[neighbor] += bin_counts[min_count_idx]
    bin_counts[min_count_idx] *= 9  # Adjust the count of the merged bin
    bin_counts[min_count_idx] //= 10  # Ensure at least 10 particles in the merged bin

# Output the final bins and their particle counts
print("Final bins:")
print(bin_counts)
