# X_split = X.reshape(num_heads, depth, m)


# Resulting X_split:
X_split = np.array([
    [
        [1, 2, 3],  # First head, first feature
        [4, 5, 6],  # First head, second feature
        [7, 8, 9],  # First head, third feature
        [1, 1, 1],  # First head, fourth feature
    ],
    [
        [2, 2, 2],  # Second head, first feature
        [3, 3, 3],  # Second head, second feature
        [4, 4, 4],  # Second head, third feature
        [5, 5, 5],  # Second head, fourth feature
    ]
])  # Shape: (2, 4, 3)



head_1_output = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
    [0.1, 0.1, 0.1],
])  # Shape: (4, 3)


head_2_output = np.array([
    [0.2, 0.2, 0.2],
    [0.3, 0.3, 0.3],
    [0.4, 0.4, 0.4],
    [0.5, 0.5, 0.5],
])  # Shape: (4, 3)


concatenated_heads = np.concatenate([head_1_output, head_2_output], axis=0)

# Resulting concatenated_heads:
concatenated_heads = np.array([
    [0.1, 0.2, 0.3],  # First head, first feature
    [0.4, 0.5, 0.6],  # First head, second feature
    [0.7, 0.8, 0.9],  # First head, third feature
    [0.1, 0.1, 0.1],  # First head, fourth feature
    [0.2, 0.2, 0.2],  # Second head, first feature
    [0.3, 0.3, 0.3],  # Second head, second feature
    [0.4, 0.4, 0.4],  # Second head, third feature
    [0.5, 0.5, 0.5],  # Second head, fourth feature
])  # Shape: (8, 3)


concatenated_heads = np.concatenate([head_1_output, head_2_output], axis=1)

# Resulting concatenated_heads:
concatenated_heads = np.array([
    [0.1, 0.2, 0.3, 0.2, 0.2, 0.2],
    [0.4, 0.5, 0.6, 0.3, 0.3, 0.3],
    [0.7, 0.8, 0.9, 0.4, 0.4, 0.4],
    [0.1, 0.1, 0.1, 0.5, 0.5, 0.5],
])  # Shape: (4, 6)
