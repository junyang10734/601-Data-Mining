import numpy as np
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
from sklearn.cluster import KMeans
from utils import *
from sklearn.decomposition import PCA

def epsilon(data, e):
    m = data.shape[0]
    n = data.shape[1]
    distance = np.zeros((m,m))
    W = np.zeros((m,m))
    for i in range(0,m):
        for j in range(0,m):
            d_tmp = 0
            for k in range(0,n):
                d_tmp = d_tmp + pow(float(data[i][k])-float(data[j][k]),2)
            distance[i,j] = d_tmp
            if distance[i,j] <= e:
                W[i,j] = np.exp(-distance[i,j]/2)
            if i == j:
                W[i,j] = 0           
    return W

def knn(data, n):
    m = data.shape[0]
    n = data.shape[1]
    distance = np.zeros((m,m))
    W = np.zeros((m,m))
    for i in range(0,m):
        for j in range(0,m):
            d_tmp = 0
            for k in range(0,n):
                d_tmp = d_tmp + pow(float(data[i][k])-float(data[j][k]),2)
            distance[i,j] = d_tmp

    for i in range(m):
        dist_with_index = zip(distance[i], range(m))
        dist_with_index = sorted(dist_with_index, key=lambda x:x[0])
        neighbours_id = [dist_with_index[k][1] for k in range(n+1)] 

        for j in neighbours_id:
            W[i][j] = np.exp(-distance[i][j]/2)
            W[j][i] = W[i][j]

    return W

def fully_connected(data, sigma):
    m = data.shape[0]
    n = data.shape[1]
    distance = np.zeros((m,m))
    W = np.zeros((m,m))
    for i in range(0,m):
        for j in range(0,m):
            d_tmp = 0
            for k in range(0,n):
                d_tmp = d_tmp + pow(float(data[i][k])-float(data[j][k]),2)
            distance[i,j] = d_tmp
            W[i,j] = np.exp(-distance[i,j]/(sigma*sigma))
            if i == j:
                W[i,j] = 0   
    return W


# draw figure
def draw(pcaData, clusters, ground = False):
    colorMark = ['red', 'blue', 'green', 'fuchsia', 'y', 'black', 'peru', 'blueviolet', 'olivedrab', 'silver',
                 'deepskyblue']
    if ground:
        plt.title('ground-truth')
    else:
        plt.title('spectral')

    # xl, yl = [], []
    x0, x1, x2, x3, x4 = [],[],[],[],[]
    y0, y1, y2, y3, y4 = [], [], [], [], []
    for i, item in enumerate(clusters):
        if item == 0:
            point = pcaData[i]  # Find the coordinates of the points after dimension reduction
            x0.append(point[0])
            y0.append(point[1])
        if item == 1:
            point = pcaData[i]  # Find the coordinates of the points after dimension reduction
            x1.append(point[0])
            y1.append(point[1])
        if item == 2:
            point = pcaData[i]  # Find the coordinates of the points after dimension reduction
            x2.append(point[0])
            y2.append(point[1])
        if item == 3:
            point = pcaData[i]  # Find the coordinates of the points after dimension reduction
            x3.append(point[0])
            y3.append(point[1])
        if item == 4:
            point = pcaData[i]  # Find the coordinates of the points after dimension reduction
            x4.append(point[0])
            y4.append(point[1])

    # for key in clusters:
    #     c = clusters[key]
    #     x, y = [], []
    #     for item in c:
    #         point = pcaData[item]  # Find the coordinates of the points after dimension reduction
    #         x.append(point[0])
    #         y.append(point[1])
    #     xl.append(x)
    #     yl.append(y)

    plt.scatter(x0, y0, s=50, label='class1', c=colorMark[0], marker='.', alpha=None,
                edgecolors='white')
    plt.scatter(x1, y1, s=50, label='class2', c=colorMark[1], marker='.', alpha=None,
                edgecolors='white')
    plt.scatter(x2, y2, s=50, label='class3', c=colorMark[2], marker='.', alpha=None,
                edgecolors='white')
    plt.scatter(x3, y3, s=50, label='class4', c=colorMark[3], marker='.', alpha=None,
                edgecolors='white')
    plt.scatter(x4, y4, s=50, label='class5', c=colorMark[4], marker='.', alpha=None,
                edgecolors='white')
    # for i in range(len(clusters)):
    #     plt.scatter(xl[i], yl[i], s=50, label='class' + str(i + 1), c=colorMark[i], marker='.', alpha=None,
    #                 edgecolors='white')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    # load file
    file_list = ['./data/cho.txt','./data/iyer.txt']
    n_clusters = [5,10]
    sigma = [1,1.75]

    for n in range(0,1):
        data_list = []
        ground_truth_list = []
        f = open(file_list[n],'r')
        lines = f.readlines()
        for line in lines:
            line_list = line.strip().split()
            valid_line_list = line_list[2:]
            label = int(line_list[1])
            data_list.append(valid_line_list)
            ground_truth_list.append(label)
        data = np.array(data_list)
        print(data)
        ground_truth = np.array(ground_truth_list)
        ground_truth = np.reshape(ground_truth, (ground_truth.shape[0],1))

        W = fully_connected(data, sigma[n])
        D = np.zeros((data.shape[0],data.shape[0]))
        D_element = np.sum(W, axis=1)
        s_D_t = 1.0/(D_element**(0.5))
        for i in range(len(s_D_t)):
            if s_D_t[i] == float("inf") or s_D_t[i] == float("-inf"):
                s_D_t[i] = 10000000000.0
        s_D = np.diag(s_D_t)

        for i in range(0,data.shape[0]):
            for j in range(0,data.shape[0]):
                D[i,i] = D[i,i] + W[i,j]
        L = D - W
        L_norm = np.dot(np.dot(s_D, L), s_D)
        print(L_norm)

        # eigenvalues, eigenvectors_tmp = LA.eig(L_norm)
        # value_index_tmp = np.argsort(eigenvalues)
        # value_index = value_index_tmp[0:n_clusters[n]]
        # eigenvectors = eigenvectors_tmp[:,value_index]
        #
        # ev_real = eigenvectors.real
        # ev_norm = np.linalg.norm(ev_real, axis=1)
        # np.seterr(divide='ignore', invalid='ignore')
        # ev_after_norm = (ev_real.T / ev_norm).T
        # for i in range(0,ev_after_norm.shape[0]):
        #     for j in range(0,ev_after_norm.shape[1]):
        #         if np.isnan(ev_after_norm[i,j]):
        #             ev_after_norm[i,j] = 0
        #
        #
        #
        # kmeans = KMeans(n_clusters=n_clusters[n])
        # results = kmeans.fit(ev_after_norm).labels_
        # print(results)
        #
        # # PCA decrease dimension
        # pca = PCA(n_components=2)
        # pcaData = pca.fit_transform(data)
        # pcaData = np.array(pcaData).tolist()
        #
        # draw(pcaData, results)
        # rand = rand_index(results, ground_truth)
        # jaccard = jaccard_coefficient(results, ground_truth)
        # print(rand)
        # print(jaccard)
               
        # visualize(data, results, str(n) + '.png')