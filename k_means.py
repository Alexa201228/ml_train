from sklearn.model_selection import train_test_split
import mglearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

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