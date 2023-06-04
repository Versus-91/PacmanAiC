import numpy as np

# Define the height and width of the matrix
height = 10
width = 10

# Create a matrix with 5 channels
matrix = np.zeros((height, width, 5))

# Access and modify the values in the matrix
matrix[:, :, 0] = 1  # Set values in the first channel to 1
matrix[:, :, 1] = 2  # Set values in the second channel to 2
matrix[:, :, 2] = 3  # Set values in the third channel to 3
matrix[:, :, 3] = 4  # Set values in the fourth channel to 4
matrix[:, :, 4] = 5  # Set values in the fifth channel to 5

# Print the matrix
print(matrix)