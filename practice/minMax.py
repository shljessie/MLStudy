import numpy as np

# Original data
X = np.array([
    [14, 25],
    [65, 33],
    [0, 996],
    [997, 997],
    [22, 60],
    [90, 0],
    [38, 10],
    [996, 100]
])

# Initialize new transformed array with additional columns for special values
newX = np.zeros((X.shape[0], 7))  # 7 columns (original 2 + 5 binary indicators)

# Copy original values, with replacements for special cases in Feature 1 and Feature 2
for i, (f1, f2) in enumerate(X):
    # Feature 1 processing
    if f1 == 996:
        newX[i, 2] = 1  # feature1_no_open_loan
        newX[i, 0] = 0  # Set original feature1 value to 0
    elif f1 == 997:
        newX[i, 3] = 1  # feature1_no_open_loan_with_invalid_date
        newX[i, 0] = 0  # Set original feature1 value to 0
    else:
        newX[i, 0] = f1  # Keep normal value for Feature 1

    # Feature 2 processing
    if f2 == 99999999996:
        newX[i, 4] = 1  # feature2_special_case_96
        newX[i, 1] = 0  # Replace with 0 or mean/median value if needed
    elif f2 == 99999999997:
        newX[i, 5] = 1  # feature2_special_case_97
        newX[i, 1] = 0  # Replace with 0 or mean/median value if needed
    elif f2 == 99999999998:
        newX[i, 6] = 1  # feature2_special_case_98
        newX[i, 1] = 0  # Replace with 0 or mean/median value if needed
    else:
        newX[i, 1] = f2  # Keep normal value for Feature 2

# Resulting transformed data
print("Transformed Data:\n", newX)


min_vals = newX[:, :2].min(axis=0)
max_vals = newX[:, :2].max(axis=0)
newX[:, :2] = (newX[:, :2] - min_vals) / (max_vals - min_vals)


print("Transformed and Normalized Data:\n", newX)




# https://colab.research.google.com/drive/1Wocm-L7Hf_VnYpSqCBFONIBFPKaygVrK#scrollTo=Y1I5Ex8H3M7l 

"""
This is the notebook I used to review numpy concepts.
"""