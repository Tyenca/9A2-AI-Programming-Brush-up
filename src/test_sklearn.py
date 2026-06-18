import sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

print("scikit-learn version:", sklearn.__version__)

X, y = load_iris(return_X_y=True)
clf = LogisticRegression(max_iter=200).fit(X, y)
print("accuracy:", clf.score(X, y))