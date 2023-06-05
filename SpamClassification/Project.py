import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore')

"""Data"""
data = pd.read_csv('spam.csv')
X = data.values[:, 0:56]
y = data.values[:, 57]

"""Classifiers"""
# knn
knn = KNeighborsClassifier(p=1,
                           n_jobs=-1,
                           leaf_size=5)

# multi-layered perceptron
mlp = MLPClassifier(activation='tanh',
                    alpha=0.05,
                    learning_rate='adaptive')

# random forest
rf = RandomForestClassifier(bootstrap=False,
                            max_depth=100,
                            n_estimators=200)

# ensemble model
evc = VotingClassifier(estimators=[('rf', rf),
                                   ('mlp', mlp)],
                       voting='hard')

"""Training"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


"""
fit the training data to the model,
store the algorithm prediction in a local variable to score the model
use the prediction from before to get the accuracy and precision and stores those in local variables
print the precision and accuracy to 2 decimal places
"""
def evaluate(model_name, model):
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    accuracy = accuracy_score(y_test, predict)
    precision = precision_score(y_test, predict)
    print('%s:' % model_name)
    print('Accuracy: {:2.2%}'.format(accuracy))
    print('Precision: {:2.2%}'.format(precision))
    print()


"""Function calls"""
evaluate('knn', knn)
evaluate('mlp', mlp)
evaluate('rf', rf)
evaluate('evc', evc)

