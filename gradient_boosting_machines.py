from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

# Overfitting
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)
print(f'Training set accuracy: {gbrt.score(X_train, y_train)}')
print(f'Test set accuracy: {gbrt.score(X_test, y_test)}')

# Fix overfitting
gbrt1 = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt1.fit(X_train, y_train)
print(f'Training set accuracy: {gbrt1.score(X_train, y_train)}')
print(f'Test set accuracy: {gbrt1.score(X_test, y_test)}')

gbrt2 = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt2.fit(X_train, y_train)
print(f'Training set accuracy: {gbrt2.score(X_train, y_train)}')
print(f'Test set accuracy: {gbrt2.score(X_test, y_test)}')


def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Важность признака")
    plt.ylabel("Признак")
    plt.show()


plot_feature_importances_cancer(gbrt)
