import numpy as np
import utils

# load dataset
def loadDataSet(filename):
    with open('./' + filename + '.txt', 'r') as f:
        data = f.readlines()
        dataSet = []
        b = {'Present': 1, 'Absent': 0}
        for i in range(len(data)):
            item = data[i].split('\t')
            item[-1] = item[-1].strip('\n')
            for j, x in enumerate(item):
                if isinstance(x, str):
                    if x in b:
                        item[j] = b[x]
            newitem = list(map(lambda x: float(x), item))
            dataSet.append(newitem)
    return dataSet


def knn(k, trainData, testData):
    test_data = np.array(testData)
    train_data = np.array(trainData)
    train_attrs = train_data[:, :-1]
    train_labels = train_data[:, -1].astype(int)
    test_attrs = test_data[:, :-1]
    test_labels = []

    for i in range(len(test_data)):
        dis_matrix = []
        for j in range(len(train_data)):
            d = np.sqrt(np.sum(np.square(test_attrs[i] - train_attrs[j])))
            dis_matrix.append(d)

        sort_idx = np.argsort(dis_matrix)
        new_idx = sort_idx[0:k]
        labels = train_labels[new_idx]
        labels_num = {}
        for item in labels:
            if item not in labels_num:
                labels_num[item] = 1
            else:
                labels_num[item] += 1

        max_label = 0
        max_num = 0
        for key in labels_num:
            if labels_num[key] > max_num:
                max_label = key
                max_num = labels_num[key]

        test_labels.append(max_label)
    return test_labels


if __name__ == "__main__":
    filename = 'project3_dataset2'
    dataSet = loadDataSet(filename)
    np.random.shuffle(dataSet)
    fold = 10
    k = 20
    acc_list, p_list, r_list, f_list = [], [], [], []

    # 10-fold cross validation
    for i in range(1, fold+1):
        print('fold ' + str(i))
        trainData, testData = utils.crossValidation(dataSet, fold, i)
        test_labels = knn(k, trainData, testData)
        true_labels = np.array(testData)[:, -1].astype(int).tolist()
        acc, p, r, f = utils.eval(true_labels, test_labels)
        acc_list.append(acc)
        p_list.append(p)
        r_list.append(r)
        f_list.append(f)
        print('acc: ' + str(acc) + ';  p: ' + str(p) + ';  r: ' + str(r) + ';  f: ' + str(f))

    final_acc = sum(acc_list)/len(acc_list)
    final_p = sum(p_list)/len(p_list)
    final_r = sum(r_list)/len(r_list)
    final_f = sum(f_list)/len(f_list)
    print('acc: ' + str(final_acc))
    print('p: ' + str(final_p))
    print('r: ' + str(final_r))
    print('f: ' + str(final_f))

    # The following code runs demo data:
    # trainData = loadDataSet('project3_dataset3_train')
    # testData = loadDataSet('project3_dataset3_test')
    # test_labels = knn(k, trainData, testData)
    # true_labels = np.array(testData)[:, -1].astype(int).tolist()
    # acc, p, r, f = utils.eval(true_labels, test_labels)
    # print(test_labels)
    # print(true_labels)
    # print('acc: ' + str(acc) + ';  p: ' + str(p) + ';  r: ' + str(r) + ';  f: ' + str(f))
