import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# load dataset
def loadDataSet(filename):
    with open('./data/' + filename + '.txt', 'r') as f:
        data = f.readlines()
        dataList = []
        for i in range(len(data)):
            item = data[i].split('\t')
            item[-1] = item[-1].strip('\n')
            newitem = list(map(lambda x: float(x), item))
            dataList.append(newitem)
    return dataList


# calculate the distance of each point to each center, and assign each point to cluster
def findCluster(centerList, dataSet):
    clusters = {}
    for item in dataSet:
        p1 = item
        minD = float('inf')
        flag = -1
        for i in range(len(centerList)):
            p2 = centerList[i]
            d = getDistance(p1, p2)
            if d < minD:
                minD = d
                flag = i
        if flag not in clusters.keys():
            clusters[flag] = []
        clusters[flag].append(item)
    return clusters


# calculate the distance between two points
def getDistance(p1, p2):
    # print(p1)
    # print(p2)
    v1 = np.array(p1[2:])
    v2 = np.array(p2[2:])
    d = np.sqrt(np.sum(np.square(v1 - v2)))
    return d


# the center of each cluster is obtained through optimization.
def findCenter(clusters):
    centerList = []
    for key in clusters.keys():
        center = np.mean(np.array(clusters[key][2:]), axis=0)
        centerList.append(center)
    return np.array(centerList).tolist()


# calculate sse clusters
def calSSE(centerList, clusters):
    sse = 0.0
    for key in clusters.keys():
        v1 = np.array(centerList[key][2:])
        sum = 0
        for i in clusters[key]:
            v2 = np.array(i[2:])
            sum += np.sum(np.square(v1 - v2))
        sse += sum
    return sse


def compare(clusters, g_clusters, nums):
    m1, m2 = np.zeros([nums, nums]), np.zeros([nums, nums])
    list1, list2 = [], []

    for key in clusters.keys():
        l = []
        for i in clusters[key]:
            l.append(int(i[0]))
        list1.append(l)

    for key in g_clusters.keys():
        l = []
        for i in g_clusters[key]:
            l.append(int(i[0]))
        list2.append(l)

    for item in list1:
        x = 0
        while x < len(item):
            y = x
            while y < len(item):
                m1[item[x]-1][item[y]-1] = 1
                m1[item[y]-1][item[x]-1] = 1
                y += 1
            x += 1

    for item in list2:
        x = 0
        while x < len(item):
            y = x
            while y < len(item):
                m2[item[x]-1][item[y]-1] = 1
                m2[item[y]-1][item[x]-1] = 1
                y += 1
            x += 1

    m00, m01, m10, m11 = 0, 0, 0, 0
    for a in range(nums):
        for b in range(nums):
            if m1[a][b] == 1 and m2[a][b] == 1:
                m11 += 1
            elif m1[a][b] == 0 and m2[a][b] == 1:
                m01 += 1
            elif m1[a][b] == 1 and m2[a][b] == 0:
                m10 += 1
            else:
                m00 += 1

    r = (m11 + m00) / (m00 + m01 + m10 + m11)
    j = m11 / (m01 + m10 + m11)
    return r, j


def draw(pcaData, clusters, ground = False):
    colorMark = ['red', 'blue', 'green', 'y', 'black', 'peru', 'blueviolet', 'fuchsia', 'olivedrab', 'silver', 'deepskyblue']
    if ground:
        plt.title('ground-truth')
    else:
        plt.title('k-means')

    xl, yl = [], []
    for key in clusters.keys():
        c = clusters[key]
        x, y = [], []
        for item in c:
            point = pcaData[int(item[0])-1]  # 根据id找到pca后的该点的坐标
            x.append(point[1])
            y.append(point[2])
        xl.append(x)
        yl.append(y)

    for i in range(len(clusters)):
        plt.scatter(xl[i], yl[i], s=50, label='class'+str(i+1), c=colorMark[i], marker='.', alpha=None, edgecolors='white')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    filename = input("please input file name: ")
    k = int(input("please input k: "))
    # k = 5
    dataSet = loadDataSet(filename)
    # 从数据集中随机选取k个数据作为质心
    centerList = random.sample(dataSet, k)

    # 将每个数据分配到相应的cluster中
    clusters = findCluster(centerList, dataSet)
    sse1 = - 0.0001
    sse2 = calSSE(centerList, clusters)
    t = 1
    while abs(sse2-sse1) >= 0.0001:
        sse1 = sse2
        centerList = findCenter(clusters)
        clusters = findCluster(centerList, dataSet)
        sse2 = calSSE(centerList, clusters)
        t = t + 1
    print(t)
    print(clusters)
    newData = []
    for item in dataSet:
        newItem = item[2:]
        newData.append(newItem)
    pca = PCA(n_components=2)
    pcaData = pca.fit_transform(newData)
    pcaData = np.array(pcaData).tolist()
    for i, item in enumerate(pcaData):
        item.insert(0, i+1)
    print(clusters)
    g_clusters = {}
    for item in dataSet:
        c = item[1]
        if c not in g_clusters.keys():
            g_clusters[c] = []
        g_clusters[c].append(item)

    # print(g_clusters)
    draw(pcaData, g_clusters, True)
    draw(pcaData, clusters)

    r, j = compare(clusters,g_clusters, len(dataSet))
    print('randIndex is:' + str(r))
    print('jc is:' + str(j))
