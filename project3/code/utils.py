import numpy as np
import csv

# dataSet: original data
# fold: number of fold
# i: order
def crossValidation(dataSet, fold, i):
    total = len(dataSet)
    test_num = int(total/fold)
    trainData, testData = [], []
    start = test_num*(i-1)
    end = start + test_num

    for i, item in enumerate(dataSet):
        if i >= start and i < end:
            testData.append(item)
        else:
            trainData.append(item)

    return trainData, testData


# label1: actual class array
# label2: predict class array
def eval(label1, label2):
    a, b, c, d = 0, 0, 0, 0
    for i in range(len(label1)):
        if label1[i] == 1 and label2[i] == 1:
            a += 1
        elif label1[i] == 1 and label2[i] == 0:
            b += 1
        elif label1[i] == 0 and label2[i] == 1:
            c += 1
        else:
            d += 1

    acc = (a+d)/(a+b+c+d)
    p = a/(a+c)
    r = a/(a+b)
    f = 2*a/(2*a+b+c)

    return acc, p, r, f


# read .csv file
def loadCSVfile(filename, hasHead=False):
    file_path = './' + filename + '.csv'
    tmp = np.loadtxt(file_path, dtype=np.str, delimiter=",")
    if hasHead:
        data = tmp[1:, 1:].astype(np.float)
        return data
    else:
        data = tmp[:, 1:].astype(np.float)
        index = tmp[:, 0].astype(np.int)
        return index, data


# write .csv file
def wirteCSVfile(filename, index, data):
    filename = './' + filename + '.csv'
    format_data = []
    for i, item in enumerate(data):
        format_data.append([index[i], item])
    with open(filename, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "label"])
        writer.writerows(format_data)
        csvfile.close()
