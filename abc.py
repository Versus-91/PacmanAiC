import math
import matplotlib.pyplot as plt

# Define the parameters
eps_start = 1.0
eps_end = 0.05
eps_decay = 100000
# Assuming counter is a range of values from 0 to 99
counter = range(500000)

# Calculate eps_threshold for each value of counter
eps_threshold = [eps_end + (eps_start - eps_end) *
                 math.exp(-1. * c / eps_decay) for c in counter]

# Plot the chart
plt.plot(counter, eps_threshold)
plt.xlabel('Counter')
plt.ylabel('Eps Threshold')
plt.title('Exponential Decay of Eps Threshold')
plt.grid(True)
plt.show()
