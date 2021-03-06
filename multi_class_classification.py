import mglearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC

X, y = make_blobs(random_state=42)


# plt.legend(['Class 0', 'Class 1', 'Class 2'])

linear_svm = LinearSVC().fit(X, y)
print(f'Coef form {linear_svm.coef_.shape}')
print(f'Const form {linear_svm.intercept_.shape}')

mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.legend(['class 0', 'class 1', 'class 2', 'class line 0', 'class line 1', 'class line 2'], loc=(1.01, 0.3))
plt.xlabel('Tag 0')
plt.ylabel('Tag 1')

plt.show()