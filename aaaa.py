import torch

# Define the number of directions
num_directions = 4  # Replace with the actual number of directions

# Define the index of the desired direction
direction_index = 2  # Replace with the actual index of the desired direction

# Create a one-hot encoded vector
one_hot_direction = torch.eye(num_directions)[direction_index]

# Example usage:
print(one_hot_direction)