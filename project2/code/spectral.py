import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# load dataSet
def loadDataSet(filename):
    with open('./' + filename + '.txt', 'r') as f:
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

    # W: similarity matrix
    W = np.zeros([nums, nums])
    for i in range(nums - 1):
        for j in range(i + 1, nums):
            d = np.sqrt(np.sum(np.square(data[i] - data[j])))
            w = np.exp(-(d ** 2 / np.square(sigma)))
            W[i][j] = w
            W[j][i] = w

    # D: degree matrix
    D = np.zeros([nums, nums])
    for i in range(nums):
        D[i][i] = sum(W[i])

    # L: Laplacian matrix
    L = D - W

    # Normalized Laplacian Matrix
    D1 = np.zeros([nums, nums])
    for i in range(nums):
        if D[i][i] != 0:
            D1[i][i] = 1 / (D[i][i]**0.5)

    L = np.dot(np.dot(D1, L), D1)

    return L


# draw figure
def draw(pcaData, clusters, filename, r, j, ground = False):
    if ground:
        plt.title("spectral of ground-truth")
    else:
        plt.title("spectral of sample " + str(filename) + "\nrand index " + str(r) + "\njaccard corfficient " + str(j))

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
        plt.scatter(xl[i], yl[i], label='class' + str(i + 1))

    plt.legend(loc='best')
    plt.show()

# compute ri and jc
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
    filename = 'cho'
    dataSet = loadDataSet(filename)
    k = 5
    sigma = 2
    data = np.array(dataSet)[:, 2:]

    # get Laplacian matrix
    L = getMatrix(data, sigma)

    # get eigen value and vector
    e_value, e_vector = np.linalg.eig(L)

    # sort eigen value and eigen vector
    sort_idx = np.argsort(e_value)
    new_k_idx = sort_idx[0:k]
    new_e_vector = e_vector[:, new_k_idx]



    # k-means cluster
    # initIds = [3, 5, 9]
    # centerList = []
    # for i in initIds:
    #     centerList.append(new_e_vector[i - 1].tolist())

    # sp_kmeans = KMeans(init=np.array(centerList), n_clusters=k, n_init=1)
    sp_kmeans = KMeans(n_clusters=k)
    sp_kmeans.fit(new_e_vector)

    clusters = {}
    for i, item in enumerate(sp_kmeans.labels_):
        if item not in clusters:
            clusters[item] = []
        clusters[item].append(i)

    # result of ground-truth
    g_clusters = {}
    for i, item in enumerate(dataSet):
        c = item[1]
        if c not in g_clusters.keys():
            g_clusters[c] = []
        g_clusters[c].append(i)

    # PCA decrease dimension
    pca = PCA(n_components=2)
    pcaData = pca.fit_transform(data)
    pcaData = np.array(pcaData).tolist()

    # calculate rand index and Jaccard Coefficient
    r, j = compare(clusters, g_clusters, len(dataSet))

    # draw figure
    # draw(pcaData, g_clusters, filename, r, j, True)
    draw(pcaData, clusters, filename, r, j)