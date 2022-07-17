import numpy as np

X = np.array([[0, 1, 0, 1],
              [1, 0, 1, 1],
              [0, 0, 0, 1],
              [1, 0, 1, 0]])

y = np.array([0, 1, 0, 1])
counts = {}


for label in np.unique(y):
    print(y == label)
    counts[label] = X[y == label].sum(axis=0)
    # Explanation:
    # first round print will show [ True False  True False]
    # which means that we look at 1 and 3 row of each column
    # second round prints [False  True False  True]
    # which means that we look at 2 and 4 row of each column
print(f'Tags frequency:\n{counts}')
# Tags frequency:
# {0: array([0, 1, 0, 2]), 1: array([2, 0, 2, 1])}
