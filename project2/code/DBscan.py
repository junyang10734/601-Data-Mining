import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class DBscan:
    def __init__(self, id=0, visited=False, pts=[], cluster=-1): # Set class attributes
        self.id = id
        self.visited = visited
        self.pts = pts
        self.cluster = cluster

def dist(x,y):
    return np.sqrt(np.sum((x-y)**2))

def dbscan(name,x,y,eps,nop):
    # Initicial cluster index and oiunt sets
    cluster = 0
    pointSet = []

    # Build distance matrix
    distMatrix = np.zeros((len(x), len(x)))
    # print(distMatrix.shape)
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            distMatrix[i, j] = dist(x[i], x[j])
            distMatrix[j, i] = distMatrix[i, j]
    print(distMatrix)

    # Count number of points within e distance
    for i in range(len(distMatrix)):
        cnt = 0
        pt = []
        for j in range(len(distMatrix[i])):
            if distMatrix[i,j] < eps:
                pt.append(j)
        pointSet.append(DBscan(i,False,pt,-1))
    # print(pointSet)

    # Start travsel each unvisited point
    for point in pointSet:
        if point.visited == False:
            point.visited = True
            if len(point.pts) >= nop:
                cluster+=1
                expandClass(pointSet, point, cluster, nop)

    # Summary cluster
    print('cluster',cluster)
    classes = [[] for i in range(cluster+1)]
    # print(classes)
    for point in pointSet:
        # print(point.id,point.cluster)
        if point.cluster == -1:
            classes[0] +=[point.id]
        else:
            classes[point.cluster]+=[point.id]
    print('classes',classes)

    # Generate predict y
    predict = np.zeros(y.shape, dtype=int)
    for j in classes[0]:
        predict[j] = -1
    for i in range(1, len(classes)):
        for j in classes[i]:
            predict[j] = i
    print(predict)

    # Calculate rand index and jaccard coefficient
    TP = TN = FP = FN = 0
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            if (predict[i] == predict[j] and y[i] == y[j]):
                TP += 1
            if (predict[i] != predict[j] and y[i] != y[j]):
                TN += 1
            if (predict[i] == predict[j] and y[i] != y[j]):
                FP += 1
            if (predict[i] != predict[j] and y[i] == y[j]):
                FN += 1
    print('rand index of', name, 'is', (TP + TN) / (TP + TN + FP + FN))
    print('jaccard coefficient of', name, 'is', TP / (TP + FP + FN))

    # Use PCA to reduce dimension
    pca = PCA(n_components=2)
    predict = pca.fit_transform(x)
    # print(predict)
    plt.title("DBscan of sample " + str(name) + "\nrand index "+str((TP + TN) / (TP + TN + FP + FN)) + "\njaccard corfficient " + str(TP / (TP + FP + FN)))
    plt.scatter(predict[classes[0], 0], predict[classes[0], 1], label='outliers')
    for i in range(1, len(classes)):
        plt.scatter(predict[classes[i], 0], predict[classes[i], 1], label='class' + str(i))
    plt.legend(loc='best')
    plt.show()

def expandClass(pointSet, point, cluster, nop):
    point.cluster = cluster
    for i in point.pts:
        if pointSet[i].visited == False:
            # cluster.append(pointSet[i].id)
            pointSet[i].visited = True
            if len(pointSet[i].pts) >= nop:
                for p in pointSet[i].pts:
                    expandClass(pointSet, pointSet[p], cluster, nop)
        if pointSet[i].cluster == -1:
            pointSet[i].cluster = cluster