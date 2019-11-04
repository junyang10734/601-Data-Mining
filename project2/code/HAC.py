import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def dist(x,y):
    return np.sqrt(np.sum((x-y)**2))

def hac(name,x, y, threshold, noc):
    # Build classes
    classes = [[i-1] for i in range(x.shape[0]+1)]
    classes[0] = []
    # print(classes)

    # Build distance matrix
    distMatrix = np.zeros((len(x),len(x)))
    # print(distMatrix.shape)
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            distMatrix[i,j] = dist(x[i],x[j])
            distMatrix[j,i] = distMatrix[i,j]
    # print(distMatrix)

    # Mark the outliers
    remove = []
    for i in range(len(distMatrix)):
        out = 1
        for j in range(len(distMatrix[i])):
            if (distMatrix[i,j] < threshold) and (i != j):
                out = 0
        if out:
            classes[0].append(i)

    if len(classes[0]) != 0:
        revclass = classes[0].copy()
        revclass.reverse()
        for i in revclass:
            del classes[i+1]
            distMatrix = np.delete(distMatrix, i, axis=0)
            distMatrix = np.delete(distMatrix, i, axis=1)

    while len(classes)-1 > noc:
        # Find the smallest distance with the index of it
        minDist = [999,0,0] # store value ,a and b
        for i in range(distMatrix.shape[0]):
            for j in range(i+1,len(distMatrix[0])):
                if distMatrix[i,j] < minDist[0]:
                    minDist = [distMatrix[i,j],i,j]

        #update matrix
        for i in range(distMatrix.shape[0]):
            distMatrix[minDist[1]][i] = min(distMatrix[minDist[1]][i], distMatrix[minDist[2]][i])
            distMatrix[i][minDist[1]] = min (distMatrix[i][minDist[1]], distMatrix[i][minDist[2]])
        distMatrix = np.delete(distMatrix, minDist[2], axis=0)
        distMatrix = np.delete(distMatrix, minDist[2], axis=1)
        # print(distMatrix.shape)

        #update classes
        # if isinstance(classes[minDist[1]], int):
        #     classes[minDist[1]] = list((classes[minDist[1]], classes[minDist[2]]))
        # else:
        #     classes[minDist[1]] = classes[minDist[1]]+[classes[minDist[2]]]
        classes[minDist[1]+1] = classes[minDist[1]+1] + classes[minDist[2]+1]
        del classes[minDist[2]+1]
        print(classes)

    # Generate predict y
    predict = np.zeros(y.shape, dtype=int)
    for j in classes[0]:
        predict[j] = -1
    for i in range(1,len(classes)):
        for j in classes[i]:
            predict[j] = i
    print(predict)

    # Calculate rand index and jaccard coefficient
    TP = TN = FP = FN = 0
    for i in range(len(y)):
        for j in range(i+1, len(y)):
            if (predict[i]==predict[j] and y[i]==y[j]):
                TP += 1
            if (predict[i]!=predict[j] and y[i]!=y[j]):
                TN += 1
            if (predict[i]==predict[j] and y[i]!=y[j]):
                FP += 1
            if (predict[i]!=predict[j] and y[i]==y[j]):
                FN += 1
    print('rand index of',name,'is',(TP+TN)/(TP+TN+FP+FN))
    print('jaccard coefficient of',name,'is',TP/(TP+FP+FN))

    # Use PCA to reduce dimension
    pca = PCA(n_components=2)
    predict = pca.fit_transform(x)
    # print(predict)
    plt.title("HAC of sample " + str(name) + "\nrand index "+str((TP + TN) / (TP + TN + FP + FN)) + "\njaccard corfficient " + str(TP / (TP + FP + FN)))
    plt.scatter(predict[classes[0], 0], predict[classes[0], 1], label='outliers')
    for i in range(1, len(classes)):
        # print(i)
        plt.scatter(predict[classes[i], 0], predict[classes[i], 1], label='class' + str(i))
    plt.legend(loc='best')
    plt.show()