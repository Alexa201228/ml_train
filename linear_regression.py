import mglearn.datasets
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)

print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))

print(f'Accuracy on training set {lr.score(X_train, y_train)}')
print(f'Accuracy on test set {lr.score(X_test, y_test)}')

# Logistic Regression

X1, y1 = mglearn.datasets.make_forge()

fig, axes = plt.subplots(1, 2, figsize=(10, 3))

for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X1, y1)
    mglearn.plots.plot_2d_separator(clf, X1, fill=False, eps=0.5, ax=ax, alpha=.7)
    mglearn.discrete_scatter(X1[:, 0], X1[:, 1], y1, ax=ax)
    ax.set_title(f'{clf.__class__.__name__}')
    ax.set_xlabel('Tag 0')
    ax.set_ylabel('Tag 1')

axes[0].legend()
plt.show()