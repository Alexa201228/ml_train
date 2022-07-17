import mglearn
import matplotlib.pyplot as plt
# import graphviz
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier #, export_graphviz

# mglearn.plots.plot_tree_progressive()
# plt.show()

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42
)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print(f'Accuracy in training set: {tree.score(X_train, y_train)}')
print(f'Accuracy in test set: {tree.score(X_test, y_test)}')

for name, score in zip(cancer['feature_names'], tree.feature_importances_):
    print(name, score)
# needs conda
# export_graphviz(tree, out_file='tree.dot', class_names=['malignant', 'benign'],
#                 feature_names=cancer.feature_names, impurity=False, filled=True)
#
#
# with open('tree.dot') as f:
#     dot_graph = f.read()
# graphviz.Source(dot_graph)
