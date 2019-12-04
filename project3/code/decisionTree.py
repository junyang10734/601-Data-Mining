import numpy as np

#Read data
dataSet = 'project3_dataset2.txt'
data=[i.rstrip().split() for i in open(dataSet, 'r').readlines()]#[:200,:]
print(len(data),len(data[0]))
print(data)

#Preprocessing
fsData = []
for line in data:
    row = []
    for value in line:
        try:
            row.append(float(value))
        except:
            row.append(value)
    fsData.append(row)

print(fsData)

fold = 10
min_size = 20
max_depth = 7
accuracies = np.zeros(shape=(fold,))
precision = np.zeros(shape=(fold,))
recall = np.zeros(shape=(fold,))
f1 = np.zeros(shape=(fold,))

# Functions to seperate testSet and labels
def crossValidation(dataSet, fold, i):
    total = len(dataSet)
    test_num = int(total/fold)
    trainData, testData = [], []
    start = test_num*i
    end = start + test_num
    
    for i in range(len(dataSet)):
        if i >= start and i < end:
            testData.append(dataSet[i])
        else:
            trainData.append(dataSet[i])

    return trainData, testData
	
def sept(dataSet):
    x=[]
    y = []
    
    for line in dataSet:
        y.append(line[-1])
        lineCopy = line.copy()
        del lineCopy[-1]
        x.append(lineCopy)
        
    return x, y

#Decision tree
def Gini(groups, classes):
    total_length = 0.
    gini = 0.
    for group in groups:
        total_length += float(len(group))
        
    for group in groups:
        if(len(group)==0):
            continue
        group_len  = float(len(group))
        score = 0.
        for i in classes:
            pr = ([x[-1] for x in group].count(i))/group_len
            score += (pr * pr)
        gini += (1-score) * (group_len/total_length)
#     print(gini)
    return gini
	
def test_split(index, value, dataSet):
    left, right = list(), list()
    if (isinstance(dataSet[0][index], float)):
        for row in dataSet:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
    else:
        for row in dataSet:
            if row[index] == value:
                left.append(row)
            else:
                right.append(row)            
    return left,right
	
def unique(dataSet):
    classes = [x[-1] for x in dataSet]
    uniqueClasses = list(set(classes))
    return uniqueClasses
	
def train(trainData,oldIndexList):
    classes = unique(trainData)
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(trainData[0])-1):
        if index in oldIndexList:
            continue
        else:
            for row in trainData:
                groups = test_split(index, row[index], trainData)
                gini = Gini(groups, classes)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
#     b_groups = deleteColumn(b_index,b_groups)
#     print(b_index)
    indexList = oldIndexList.copy()
    indexList.append(b_index)
    print(indexList)
    return {'index':b_index, 'indexList':indexList, 'val':b_value, 'gini':b_score, 'groups':b_groups}
	
# Create a terminal node value
def stopDivide(group):
    classes = [x[-1] for x in group]
    return max(np.unique(classes), key = list(classes).count)


def splitNode(node, min_size, max_depth, depth):
#     print(node['groups'])
    left, right = node['groups']
    indexList = node['indexList']
    
    print('depth',depth)

    if(len(left)==0) or (len(right)==0):
        classes = []
        if len(left)==0:
            node['left'] = node['right'] = stopDivide(right)
            return
        else:
            node['left'] = node['right'] = stopDivide(left)
            return
    
#     if((left.size==0) or (right.size==0)):
#         classes = list()
#         if(left.size==0):
#             classes = right[:,-1]  
#         else:
#             classes = left[:,-1]
            
#         node['left'] = node['right'] = max(np.unique(classes), key = list(classes).count)
#         return
    
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = stopDivide(left), stopDivide(right)
        return
    
    if (len(left)<=min_size):
        node['left'] =  stopDivide(left)
    else:
        node['left'] = train(left,indexList)
        splitNode(node['left'],min_size, max_depth, depth+1)
    
    if(len(right)<=min_size):
        node['right'] = stopDivide(right)
    else:
        node['right'] = train(right,indexList)
        splitNode(node['right'],min_size, max_depth, depth+1)
		
def predict(node,row):
    if (isinstance( node['val'], float)):
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
    else:
        if row[node['index']] == node['val']:
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
        if (isinstance(node['val'], float)):
            print(depth*'- ', 'Column', (node['index']),'<',node['val'],'gini:',node['gini'])
        else:
            print(depth*'- ', 'Column', (node['index']),'=',node['val'],'gini:',node['gini'])
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print(depth*'- ', node)


def decisionTree(trainData,testData):
#     x_train,y_train = sept(trainData)
    x_test,y_test = sept(testData)
    root = train(trainData,[])
    splitNode(root, min_size, max_depth, 1)
    
    print_tree(root)
    
    pred=[]
    for row in x_test:
        pred.append(predict(root,row))
#     np.asarray(pred)
    return y_test,pred

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

#Main function
for i in range(fold):
    trainData, testData = crossValidation(fsData,fold,i)
    print("Fold",i,"start")
    true_test,pred_test = decisionTree(trainData, testData)
    print('true',true_test)
    print('pred',pred_test)
    confusion(true_test,pred_test,i)
    
print("Average accuracy: " + str(accuracies.mean()))
print("Average precision: " + str(precision.mean()))
print("Average recall: " + str(recall.mean()))
print("Average F1-measure: " + str(f1.mean()))