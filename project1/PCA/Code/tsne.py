import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def tSNE_impl(file_name):
    #read the last column of the file as disease list
    file_handle = open(file_name + ".txt", "r")
    lines = file_handle.readlines()
    num_cols = len(lines[0].strip().split("\t"))

    disease_list = []

    for line in lines:
        disease_list.append(line.strip().split("\t")[-1])

    #encode the disease list assigning a unique number to each disease in order to color the final plot
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

    #read the input features into numpy array
    input_matrix = np.loadtxt(file_name + ".txt", delimiter="\t", usecols = range(num_cols - 1))



    # plot diseases using the reduced matrix as the coordinates
    colors = [plt.cm.jet(float(i) / max(UniqueDiseaseEncodedList)) for i in UniqueDiseaseEncodedList]
    filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    tsne_results=get_tSNE(input_matrix)
    for i, u in enumerate(UniqueDiseaseList):
        xi = [p for (j,p) in enumerate(tsne_results[:,0]) if disease_list[j] == u]
        yi = [p for (j,p) in enumerate(tsne_results[:,1]) if disease_list[j] == u]
        plt.scatter(xi, yi, c=colors[i], marker=filled_markers[i], label=str(u))

    plt.title(file_name + " tSNE scatter plot")
    plt.legend()
    plt.show()

def get_tSNE(input_matrix):
    tsne = TSNE(n_components=2, n_iter=600)
    tsne_results = tsne.fit_transform(input_matrix)
    return tsne_results

#tSNE_impl("pca_a")
#tSNE_impl("pca_b")
#tSNE_impl("pca_c")
tSNE_impl("pca_demo")