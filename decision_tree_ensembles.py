from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn
import numpy as np

X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    ax.set_title(f'Tree {i}')
    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)

mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1], alpha=.4)
axes[-1, -1].set_title('Random forest')
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

cancer = load_breast_cancer()


def plot_feature_importance(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel('Tag importance')
    plt.ylabel('Tag')


X1_train, X1_test, y1_train, y1_test = train_test_split(
    cancer.data, cancer.target, random_state=0
)
forest1 = RandomForestClassifier(n_estimators=100, random_state=0)
forest1.fit(X1_train, y1_train)
plot_feature_importance(forest1)
plt.show()