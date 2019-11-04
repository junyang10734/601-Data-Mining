import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from HAC import hac
from DBscan import dbscan

# Load data
def loadDataSet():
    with open('cho.txt', 'r') as f1:
        data = f1.readlines()
        cho = []
        for i in range(len(data)):
            data[i] = data[i].strip('\n')
            item = data[i].split('\t')
            item = list(map(float, item))
            cho.append(item[1:])
    with open('iyer.txt', 'r') as f2:
        data = f2.readlines()
        iyer = []
        for i in range(len(data)):
            data[i] = data[i].strip('\n')
            item = data[i].split('\t')
            item = list(map(float, item))
            iyer.append(item[1:])
    return np.array(cho),np.array(iyer)  # return two list of lists

def normalize(x): #Central normalization
    xNorm = (x - x.mean()) / (x.max() - x.min()) # Normalize the dataset with the mean normalization equation
    return xNorm




# Load the data
cho, iyer = loadDataSet()

#split the data to x and y
y_cho = cho[:,0].astype(int)
x_cho = cho[:,1:]
y_iyer = iyer[:,0].astype(int)
x_iyer = iyer[:,1:]
# print(y_cho,x_cho,y_iyer,x_iyer)

# normalize the data
# x_cho = normalize(x_cho)
# x_iyer = normalize(x_iyer)

# Plot classification with ground truth
pca = PCA(n_components=2)
gt = pca.fit_transform(x_cho)
plt.title("Ground truth of sample cho")
# plt.scatter(gt[y_cho==1, 0], gt[y_cho==1, 1],label='class1')
# plt.scatter(gt[y_cho==2, 0], gt[y_cho==2, 1],label='class2')
# plt.scatter(gt[y_cho==3, 0], gt[y_cho==3, 1],label='class3')
# plt.scatter(gt[y_cho==4, 0], gt[y_cho==4, 1],label='class4')
# plt.scatter(gt[y_cho==5, 0], gt[y_cho==5, 1],label='class5')
plt.scatter(gt[y_cho==-1, 0], gt[y_cho==-1, 1], label='outliers')
for i in range(1, max(y_cho)+1):
    plt.scatter(gt[y_cho==i, 0], gt[y_cho==i, 1], label='class' + str(i))
plt.legend(loc='best')
plt.show()

gt = pca.fit_transform(x_iyer)
plt.title("Ground truth of sample iyer")
plt.scatter(gt[y_iyer==-1, 0], gt[y_iyer==-1, 1], label='outliers')
for i in range(1, max(y_iyer)+1):
    plt.scatter(gt[y_iyer==i, 0], gt[y_iyer==i, 1], label='class' + str(i))
plt.legend(loc='best')
plt.show()


# Custer with Hierarchical Agglomerative clustering with Min approach
noc = 5
threshold = 4
hac('cho', x_cho,y_cho, threshold, noc)

noc = 10
threshold = 4
hac('iyer', x_iyer,y_iyer, threshold, noc)

# Cluster with DBscan
e = 1.08
p = 4
dbscan('cho',x_cho,y_cho,e,p)

e = 1
p = 5
dbscan('iyer',x_iyer,y_iyer,e,p)
