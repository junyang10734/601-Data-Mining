from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import utils


if __name__ == '__main__':
    train_id, train_features = utils.loadCSVfile('train_features')
    train_label = utils.loadCSVfile('train_label', True)
    test_id, test_features = utils.loadCSVfile('test_features')

    # AdaBoost
    AB = AdaBoostClassifier(n_estimators=750)
    AB.fit(train_features, train_label.ravel())
    pred1 = AB.predict(test_features).astype(int)
    utils.wirteCSVfile('adaboost-result', test_id, pred1)

    # bagging
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
    bag = BaggingClassifier(base_estimator=tree, n_estimators=100, max_samples=1.0, max_features=1.0, bootstrap=True,
                            bootstrap_features=False, n_jobs=1, random_state=1)
    bag = bag.fit(train_features, train_label.ravel())
    pred2 = bag.predict(test_features).astype(int)
    utils.wirteCSVfile('bagging-result', test_id, pred2)

    # decision tree
    clf = DecisionTreeClassifier(splitter='random', max_depth=10)
    clf = clf.fit(train_features, train_label.ravel())
    pred3 = clf.predict(test_features).astype(int)
    utils.wirteCSVfile('dt-result', test_id, pred3)

    # knn
    knn = KNeighborsClassifier(n_neighbors=60)
    knn.fit(train_features, train_label.ravel())
    pred4 = knn.predict(test_features).astype(int)
    utils.wirteCSVfile('knn-result', test_id, pred4)

    # logistic regression
    clf = LogisticRegression(solver='liblinear')
    clf.fit(train_features, train_label.ravel())
    pred5 = clf.predict(test_features).astype(int)
    utils.wirteCSVfile('lg-result', test_id, pred5)

    # Naïve Bayes
    gnb = GaussianNB()
    gnb.fit(train_features, train_label.ravel())
    pred6 = gnb.predict(test_features).astype(int)
    utils.wirteCSVfile('nb-result', test_id, pred6)

    # random forest
    rf = RandomForestClassifier(n_estimators=1000, criterion='gini', max_features='sqrt', max_depth=None,
                                min_samples_split=2, bootstrap=True, n_jobs=1, random_state=1)
    rf = rf.fit(train_features, train_label.ravel())
    pred7 = rf.predict(test_features).astype(int)
    utils.wirteCSVfile('rf-result', test_id, pred7)

    # svm
    clf = svm.SVC(gamma='auto')
    clf.fit(train_features, train_label.ravel())
    pred8 = clf.predict(test_features).astype(int)
    utils.wirteCSVfile('svm-result', test_id, pred8)
