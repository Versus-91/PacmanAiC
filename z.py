import matplotlib.pyplot as plt
import numpy as np


# def plot_epsilon_decay(epsilon_start, epsilon_end, epsilon_decay, num_episodes):
#     epsilon_values = []
#     for episode in range(num_episodes):
#         epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
#             np.exp(-episode / epsilon_decay)
#         epsilon_values.append(epsilon)

#     plt.plot(range(num_episodes), epsilon_values)
#     plt.xlabel("Episode")
#     plt.ylabel("Epsilon Value")
#     plt.title("Epsilon Decay")
#     plt.show()


# epsilon_start = 1.0
# epsilon_end = 0.1
# epsilon_decay = 500
# num_episodes = 1000

# plot_epsilon_decay(epsilon_start, epsilon_end, epsilon_decay, num_episodes)
# import matplotlib.pyplot as plt
# import math


# def plot_epsilon_decay(epsilon_start, epsilon_end, epsilon_decay, max_steps):
#     epsilon_values = []
#     for step in range(max_steps):
#         epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
#             math.exp(-1. * step / epsilon_decay)
#         epsilon_values.append(epsilon)

#     plt.plot(range(max_steps), epsilon_values)
#     plt.xlabel("Step")
#     plt.ylabel("Epsilon Value")
#     plt.title("Epsilon Decay")
#     plt.show()


# # Parameters for epsilon decay
# epsilon_start = 1.0
# epsilon_end = 0.1
# epsilon_decay = 5000
# max_steps = 100000

# Call the plot_epsilon_decay function with the desired parameters
# plot_epsilon_decay(epsilon_start, epsilon_end, epsilon_decay, max_steps)

import matplotlib.pyplot as plt
import random

EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 400000  # Adjust the decay rate as needed


def plot_epsilon_decay(eps_start, eps_end, eps_decay, max_steps):
    epsilon_values = []
    for step in range(max_steps):
        epsilon = max(eps_end, eps_start -
                      (eps_start - eps_end) * step / eps_decay)
        epsilon_values.append(epsilon)

    plt.plot(range(max_steps), epsilon_values)
    plt.xlabel("Step")
    plt.ylabel("Epsilon Value")
    plt.title("Epsilon Decay")
    plt.show()


# Parameters for epsilon decay
eps_start = EPS_START
eps_end = EPS_END
eps_decay = EPS_DECAY
max_steps = 16000

# Call the plot_epsilon_decay function with the desired parameters
plot_epsilon_decay(eps_start, eps_end, eps_decay, max_steps)
