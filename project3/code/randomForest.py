import numpy as np
import random

#Read data
dataSet = 'project3_dataset1.txt'
data=np.array([i.rstrip().split() for i in open(dataSet, 'r').readlines()])#[:200,:]
print(data.shape)
print(data)

#Preprocessing
for i in range(len(data[0])):
    column = data[:,i].copy().T
    try:
        column = column.astype(float)
    except:
        pass
    classes = np.unique(column)
    if(type(column[0])==np.str_):
        for j in range(len(classes)):
            column[column == classes[j]] = j
    data[:,i] = column.T
data=data.astype(float)
print(data[:10,3:6])

fold = 10
min_size = 1
num_features = int((len(data[0])-1)**0.5)
max_depth = num_features-1
numTrees = 3
accuracies = np.zeros(shape=(fold,))
precision = np.zeros(shape=(fold,))
recall = np.zeros(shape=(fold,))
f1 = np.zeros(shape=(fold,))


def crossValidation(dataSet, fold, i):
    total = len(dataSet)
    test_num = int(total / fold)
    trainData, testData = [], []
    start = test_num * i
    end = start + test_num

    testData = dataSet[start:end, :]
    trainData = np.append(dataSet[:start, :], dataSet[end:, :], axis=0)

    return trainData, testData
def sept(dataSet):
    train = dataSet[:,:-1]
    test = dataSet[:,-1]
    return train, test

#Random forest
def getSample(data, sample_size):
    sample = list()
    while(len(sample)<sample_size):
        i = random.randrange(len(data))
        sample.append(data[i])
    return np.array(sample)
def Gini(groups, classes):
    total_length = 0.
    gini = 0.
    for group in groups:
        total_length += float(len(group))

    for group in groups:
        if (len(group) == 0):
            continue
        group_len = float(len(group))
        score = 0.
        for i in classes:
            pr = (list(group[:, -1]).count(i)) / group_len
            score += (pr * pr)
        gini += (1 - score) * (group_len / total_length)
    return gini
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return np.array(left),np.array(right)


def train(trainData, oldIndexList, num_features):
    classes = np.unique(trainData[:, -1])
    b_index, b_value, b_score, b_groups = -1, float("inf"), float("inf"), None
    features = []

    while len(features) < num_features:
        colnum = random.randrange(len(trainData[0]) - 1)
        if colnum not in features:
            features.append(colnum)

    for col in features:
        if col in oldIndexList:
            continue
        else:
            for row in trainData:
                groups = test_split(col, row[col], trainData)
                gini = Gini(groups, classes)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = col, row[col], gini, groups
    #                 else:
    #                     print(gini)
    if b_index == -1:
        return {'index': None, 'indexList': oldIndexList, 'val': b_value, 'groups': [np.array(trainData), np.array([])]}
    indexList = oldIndexList.copy()
    indexList.append(b_index)
    print(indexList)
    #     print(b_groups)
    return {'index': b_index, 'indexList': indexList, 'val': b_value, 'groups': b_groups}
# Create a terminal node value
def stopDivide(group):
    classes = group[:,-1]
    return max(np.unique(classes), key = list(classes).count)


def splitNode(node, min_size, max_depth, depth, num_features):
    left, right = node['groups']
    indexList = node['indexList']

    print('depth', depth)

    if ((left.size == 0) or (right.size == 0)):
        classes = []
        if (left.size == 0):
            node['left'] = node['right'] = stopDivide(right)
            return
        else:
            node['left'] = node['right'] = stopDivide(left)
            return

    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = stopDivide(left), stopDivide(right)
        return

    if (len(left) <= min_size):
        node['left'] = stopDivide(left)
    else:
        #         print('left',left)
        node['left'] = train(left, indexList, num_features)
        #         print('nodeleft',node['left'])
        splitNode(node['left'], min_size, max_depth, depth + 1, num_features)
    if (len(right) <= min_size):
        node['right'] = stopDivide(right)
    else:
        node['right'] = train(right, indexList, num_features)
        splitNode(node['right'], min_size, max_depth, depth + 1, num_features)
def predict(node,row):
    if row[node['index']] < node['val']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print(depth*'- ', 'Column', (node['index']),'<',node['val'])
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print(depth*'- ', node)


def randomForest(trainData, testData, num_trees):
    forest = []
    sample_size = int(len(trainData) / num_trees)
    for i in range(0, num_trees):
        sample = getSample(trainData, sample_size)
        root = train(trainData, [], num_features)
        splitNode(root, min_size, max_depth, 1, num_features)
        forest.append(root)
        print()
        print_tree(root)

    x_test, y_test = sept(testData)
    pred = []
    for row in x_test:
        cur_pred = [predict(tree, row) for tree in forest]
        pred.append(max(np.unique(cur_pred), key=list(cur_pred).count))
    #     np.asarray(pred)
    return y_test, pred

#Confusion matrix
def confusion(true, pred, k):
    # Calculate rand index and jaccard coefficient
    TP = TN = FP = FN = 0
    for i in range(len(true)):
        for j in range(i + 1, len(true)):
            if true[i] == 1 and pred[i] == 1:
                TP += 1
            if true[i] == 0 and pred[i] == 0:
                TN += 1
            if true[i] == 0 and pred[i] == 1:
                FP += 1
            if true[i] == 1 and pred[i] == 0:
                FN += 1
    accuracies[k] = (TP + TN) / (TP + FP + FN + TN)
    precision[k] = TP / (TP + FP)
    recall[k] = TP / (TP + FN)
    f1[k] = 2 * (recall[k] * precision[k]) / (recall[k] + precision[k])

    print("fold", k, "finsh")
    print("accuracy: " + str(accuracies[k]))
    print("precision: " + str(precision[k]))
    print("recall: " + str(recall[k]))
    print("F1-measure: " + str(f1[k]))
    print()

#main
for i in range(fold):
    trainData, testData = crossValidation(data, fold, i)
    print("Fold", i, "start")
    true_test, pred_test = randomForest(trainData, testData, numTrees)
    print('true', true_test)
    print('pred', pred_test)
    confusion(true_test, pred_test, i)

print("Average accuracy: " + str(accuracies.mean()))
print("Average precision: " + str(precision.mean()))
print("Average recall: " + str(recall.mean()))
print("Average F1-measure: " + str(f1.mean()))