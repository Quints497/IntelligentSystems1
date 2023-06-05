import warnings
import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# processing the data
data = pd.read_csv("spam.csv")
X = data.values[:, 0:56]
y = data.values[:, 57]

# splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# classifiers
mlp = MLPClassifier()
knn = KNeighborsClassifier()
rf = RandomForestClassifier()
dt = DecisionTreeClassifier()
nb = GaussianNB()
svm = SVC()


# tuning hyper parameters
model_params = {
                # multi-layered perceptron
                'mlp': {
                    'model': MLPClassifier(),
                    'params': {
                        'hidden_layer_sizes': [(100,), (50, 50, 50), (50, 100, 50)],
                        'activation': ['tanh'],
                        'solver': ['sgd', 'adam', 'lbfgs'],
                        'alpha': [0.0001, 0.05, 0.1],
                        'learning_rate': ['constant', 'adaptive']
                    }
                },

                # k-nearest neighbors
                'knn': {
                    'model': KNeighborsClassifier(),
                    'params': {
                        'n_neighbors': [5, 10, 20, 30],
                        'leaf_size': [5, 10, 20, 30],
                        'n_jobs': [-1, 1],
                        'p': [1, 2]
                    }
                },

                # random forest
                'rf': {
                    'model': RandomForestClassifier(),
                    'params': {
                        'bootstrap': [True, False],
                        'max_depth': [10, 50, 100],
                        'max_features': ['auto', 'sqrt'],
                        'min_samples_leaf': [1, 2, 4],
                        'min_samples_split': [2, 5, 10],
                        "n_estimators": [200, 1100, 2000]
                    }
                },

                # decision tree
                'dt': {
                    'model': DecisionTreeClassifier(),
                    'params': {
                        'criterion': ['gini', 'entropy'],
                        'max_depth': [0, 1, 2, 3, 4, 5],
                        'min_samples_split': [2, 4, 6, 8, 10],
                        'min_samples_leaf': [1, 3, 5, 7, 9],
                    }
                },

                # support vector machine
                'svm': {
                    'model': SVC(),
                    'params': {
                        'kernel': ['rbf'],
                        'gamma': [1e-3, 1e-4],
                        'c': [1, 10, 100, 1000]
                    }
                }

}

# finding the best parameters for the classifiers
for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5)
    clf.fit(X_train, y_train)
    print()
    print('model: ', model_name)
    print('params: ', clf.best_params_)
    print('best score: ', clf.best_score_)
    print('best estimator: ', clf.best_estimator_)
    print()
    if model_name == 'mlp':
        mlp = clf.best_estimator_
    if model_name == 'knn':
        knn = clf.best_estimator_
    if model_name == 'rf':
        rf = clf.best_estimator_
    if model_name == 'dt':
        dt = clf.best_estimator_
    if model_name == 'svm':
        svm = clf.best_estimator_

# creating the voting classifier
evc = VotingClassifier(estimators=[('mlp', mlp),
                                   ('rf', rf)],
                       voting='hard')


"""
fits the training sets to the model,
gets the model prediction,
prints the accuracy and precision score for the model,
prints the classification report
"""

def evaluate(name, model):
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    print('%s:' % name)
    print('Accuracy:', accuracy_score(y_test, predict))
    print('Precision:', precision_score(y_test, predict))
    print('Classification report: \n', classification_report(y_test, predict))


# evaluating each model
evaluate('knn', knn)
evaluate('mlp', mlp)
evaluate('rf', rf)
evaluate('dt', dt)
evaluate('svm', svm)
evaluate('nb', nb)
evaluate('evc', evc)
