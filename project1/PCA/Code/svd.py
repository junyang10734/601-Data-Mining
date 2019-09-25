import numpy as np
import matplotlib.pyplot as plt

#text_file = 'pca_a.txt'
#text_file = 'pca_b.txt'
#text_file = 'pca_c.txt'
text_file = 'pca_demo.txt'

file_values = []
with open(text_file,'r') as f:
	for line in f:
		values = line.split('\t')
		file_values.append(values)

numrows = len(file_values)
numcols = len(file_values[0])

X=np.array(file_values)

labels=X[:,numcols-1]
labelset=set(labels)

X=np.delete(X,numcols-1,1)
X=np.array(X).astype(np.float)


Y = np.mean(X, axis=0)
Z=np.subtract(X,Y)


u,s,v=np.linalg.svd(Z.T)
principal_components_svd=u[:,[0,1]]

result=X.dot(principal_components_svd)

f, ax = plt.subplots()
for name in labelset:
	x = result[labels[:]==name,0]
	y = result[labels[:]==name,1]
	ax.scatter(x, y,label=name)
plt.title("SVD plot for : "+str(text_file))
plt.legend(loc='upper left',ncol=1,fontsize=12)
plt.show()