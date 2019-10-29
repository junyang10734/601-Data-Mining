import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# load dataSet
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


def getMatrix(data, sigma=1.0):
    nums = len(data)

    # M: disrance matrix
    M = np.zeros((nums, nums))
    for i in range(nums):
        for j in range(i+1, nums):
            d = np.sqrt(np.sum(np.square(data[i] - data[j])))
            M[i][j] = d  # 欧式距离
            M[j][i] = M[i][j]

    # W: similarity matrix
    W = np.zeros([nums, nums])
    for i in range(nums - 1):
        for j in range(i + 1, nums):
            w = np.exp(-(M[i][j] / np.square(sigma)))
            W[i][j] = w
            W[j][i] = w

    # D: degree matrix
    D = np.zeros([nums, nums])
    for i in range(nums):
        D[i][i] = sum(W[i])

    # L: Laplacian matrix
    # L = np.zeros([nums, nums])
    L = D - W

    return L



def getDistanceMatrix(X):
    X = np.array(X)
    nums = len(X)
    S = np.zeros((nums, nums))
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            d = np.sqrt(np.sum(np.square(X[i] - X[j])))
            S[i][j] = d  # 欧式距离
            S[j][i] = S[i][j]
    return S


def getAMatrix(M, sigma=1.0):
    nums = len(M)
    # print(nums)
    A = np.zeros([nums, nums])

    for i in range(nums-1):
        for j in range(i+1, nums):
            d = M[i][j]
            w = np.exp(-(d / np.square(sigma)))
            A[i][j] = w
            A[j][i] = w

    return A


def getLPMatrix(M):
    nums = len(M)
    D = np.zeros([nums, nums])
    L = np.zeros([nums, nums])
    for i in range(nums):
        D[i][i] = sum(M[i])

    L = D - M
    return L


def draw(pcaData, clusters, ground = False):
    colorMark = ['red', 'blue', 'green', 'fuchsia', 'y', 'black', 'peru', 'blueviolet', 'olivedrab', 'silver',
                 'deepskyblue']
    if ground:
        plt.title('ground-truth')
    else:
        plt.title('spectral')

    xl, yl = [], []
    for key in clusters.keys():
        c = clusters[key]
        x, y = [], []
        for item in c:
            point = pcaData[item]  # Find the coordinates of the points after dimension reduction
            x.append(point[0])
            y.append(point[1])
        xl.append(x)
        yl.append(y)

    for i in range(len(clusters)):
        plt.scatter(xl[i], yl[i], s=50, label='class' + str(i + 1), c=colorMark[i], marker='.', alpha=None,
                    edgecolors='white')

    plt.legend()
    plt.show()


def compare(clusters, g_clusters, nums):
    m1, m2 = np.zeros([nums, nums]), np.zeros([nums, nums])
    list1, list2 = [], []

    for key in clusters.keys():
        l = []
        for i in clusters[key]:
            l.append(i)
        list1.append(l)

    for key in g_clusters.keys():
        l = []
        for i in g_clusters[key]:
            l.append(i)
        list2.append(l)

    for item in list1:
        x = 0
        while x < len(item):
            y = x
            while y < len(item):
                m1[item[x]][item[y]] = 1
                m1[item[y]][item[x]] = 1
                y += 1
            x += 1

    for item in list2:
        x = 0
        while x < len(item):
            y = x
            while y < len(item):
                m2[item[x]][item[y]] = 1
                m2[item[y]][item[x]] = 1
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


if __name__ == '__main__':
    dataSet = loadDataSet('cho')
    k = 5
    sigma = 0.3
    # sigma = 0.4
    data = np.array(dataSet)[:, 2:]
    # label = np.array(dataSet)[:, 1]

    # get Laplacian matrix
    L = getMatrix(data, sigma)


    # 根据欧氏距离计算距离矩阵
    # similarity = getDistanceMatrix(data)
    # # 根据高斯kernel计算邻接矩阵
    # Adjacent = getAMatrix(similarity, sigma)
    # # 拉普拉斯矩阵
    # S = getLPMatrix(Adjacent)


    # get eigen value and vector
    # x, V = np.linalg.eig(L)
    e_value, e_vector = np.linalg.eig(L)

    # sort eigen value and eigen vector
    sort_idx = np.argsort(e_value)
    new_e_value = e_value[sort_idx]
    new_e_vector = e_vector[sort_idx]

    # # 将特征值排序
    # x_rand = np.argsort(x)
    # new_evalue = x[x_rand]
    # new_evector = V[x_rand]

    # get the number of columns of the feature vector
    # my_k1 = np.array(new_e_value[:-1])
    # my_k2 = np.array(new_e_value[1:])
    # gap_lam = my_k2 - my_k1
    # my_k = gap_lam.argmax() + 2
    # print(my_k)

    # get the number of columns of the feature vector
    max_gap = 0
    c = 0
    for i in range(len(new_e_value)-1):
        if new_e_value[i+1] - new_e_value[i] > max_gap:
            c = i + 2

    # k-means cluster
    sp_kmeans = KMeans(init='k-means++', n_clusters=k)
    sp_kmeans.fit(new_e_vector[:c].T)
    # print(len(sp_kmeans.labels_))
    # result of cluster
    clusters = {}
    for i, item in enumerate(sp_kmeans.labels_):
        if item not in clusters:
            clusters[item] = []
        clusters[item].append(i)
    # print(clusters)

    # result of ground-truth
    g_clusters = {}
    for i, item in enumerate(dataSet):
        c = item[1]
        if c not in g_clusters.keys():
            g_clusters[c] = []
        g_clusters[c].append(i)
    # print(g_clusters)

    # PCA decrease dimension
    pca = PCA(n_components=2)
    pcaData = pca.fit_transform(data)
    pcaData = np.array(pcaData).tolist()

    # draw figure
    draw(pcaData, g_clusters, True)
    draw(pcaData, clusters)

    # calculate rand index and Jaccard Coefficient
    r, j = compare(clusters, g_clusters, len(dataSet))
    print('randIndex is:' + str(r))
    print('jc is:' + str(j))