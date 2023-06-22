import math
import matplotlib.pyplot as plt

eps_start = 0.9 # Initial value of epsilon
eps_end = 0.05  # Final value of epsilon
eps_decay = 100000  # Decay factor for epsilon
counter = range(100000)  # Counter values

epsilon_values = [eps_end + (eps_start - eps_end) * math.exp(-1. * c / eps_decay) for c in counter]

plt.plot(counter, epsilon_values)
plt.xlabel('Counter')
plt.ylabel('Epsilon')
plt.title('Epsilon Decay')
plt.grid(True)
plt.show()