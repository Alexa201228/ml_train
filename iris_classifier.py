from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import mglearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier



iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0
)


iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
                        hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make prediction for new flower
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
print(prediction)

# Make prediction for test set and calculate accuracy

y_pred = knn.predict(X_test)
print(y_pred)
print(f'Accuracy {np.mean(y_pred == y_test)}')
# print(f'Ключи iris_dataset: {iris_dataset["target_names"]}')

# Show scatter matrix
# plt.show()
# mglearn.plots.plot_knn_classification(n_neighbors=1)

X, y = mglearn.datasets.make_forge()

x_train, x_test, Y_train, Y_test = train_test_split(X, y, random_state=0)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train, Y_train)
print(f'Predictions on test set {clf.predict(x_test)}')
print(f'Predictions accuracy {clf.score(x_test, Y_test)}')

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbours, ax in zip([1, 3, 9], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbours).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title(f'Neighbors count: {n_neighbours}')
    ax.set_xlabel('Tag 0')
    ax.set_ylabel('Tag 1')
axes[0].legend(loc=3)
plt.show()