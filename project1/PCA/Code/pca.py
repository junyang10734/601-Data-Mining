import numpy as np
import matplotlib.pyplot as plt


def pca(file_name):
    # 将最后一行读入为病症
    file_handle = open(file_name + ".txt", "r")
    lines = file_handle.readlines()
    num_cols = len(lines[0].strip().split("\t"))

    disease_list = []

    for line in lines:
        disease_list.append(line.strip().split("\t")[-1])

    # 编码不同病症对应不用颜色
    d = dict()
    counter = 0
    for disease in disease_list:
        if disease in d:
            continue
        d[disease] = counter
        counter += 1
    DiseaseEncoded = [d[disease] for disease in disease_list]

    # 创建病症list
    UniqueDiseaseSet = set(disease_list)
    UniqueDiseaseList = list(UniqueDiseaseSet)

    # list编码
    UniqueDiseaseEncodedSet = set(DiseaseEncoded)
    UniqueDiseaseEncodedList = list(UniqueDiseaseEncodedSet)

    # read the input features into numpy array
    InputMatrix = np.loadtxt(file_name + ".txt", delimiter="\t", usecols = range(num_cols - 1))

    # plot diseases using the principle components as the coordinates
    colors = [plt.cm.jet(float(i) / max(UniqueDiseaseEncodedList)) for i in UniqueDiseaseEncodedList]
    filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    principle_components_matrix=get_pcm(InputMatrix)
    for i, u in enumerate(UniqueDiseaseList):
        xi = [p for (j,p) in enumerate(principle_components_matrix[:,0]) if disease_list[j] == u]
        yi = [p for (j,p) in enumerate(principle_components_matrix[:,1]) if disease_list[j] == u]
        plt.scatter(xi, yi, c=colors[i], marker=filled_markers[i],label=str(u))

    plt.title(file_name + " scatter plot")
    plt.legend()
    plt.show()

def get_pcm(input_matrix):
    # find adjusted_matrix
    adjusted_matrix = input_matrix - input_matrix.mean(axis=0)

    # covariance
    # cov_matrix = np.dot(adjusted_matrix.T, adjusted_matrix) / adjusted_matrix.shape[0]
    cov_matrix = np.cov(adjusted_matrix.T)

    # obtain eigen values and eigen vectors of covariance matrix
    eig_val, eig_vec = np.linalg.eig(cov_matrix)

    # select top n eigen values as the principal components
    top_2_eig_val_indexes = eig_val.argsort()[-2:][::-1]
    top_2_eig_vec = eig_vec[:, top_2_eig_val_indexes]
    principle_components_matrix = np.empty([input_matrix.shape[0], top_2_eig_vec.shape[1]])
    i = 0
    for top_eig_vec in top_2_eig_vec.T:
        principle_components_matrix[:, i] = np.dot(adjusted_matrix, top_eig_vec.T)
        i += 1
    return principle_components_matrix

#pca("pca_a")
#pca("pca_b")
#pca("pca_c")
pca("pca_demo")

