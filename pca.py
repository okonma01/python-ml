import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

# generate random data-set
genes = ['gene' + str(i) for i in range(1, 101)]

wt = ['wt' + str(i) for i in range(1, 6)]
ko = ['ko' + str(i) for i in range(1, 6)]

data = pd.DataFrame(columns=[*wt, *ko], index=genes)

for gene in data.index:
    data.loc[gene, 'wt1':'wt5'] = np.random.poisson(lam=rd.randrange(10, 1000), size=5)
    data.loc[gene, 'ko1':'ko5'] = np.random.poisson(lam=rd.randrange(10, 1000), size=5)

print(data.head())
print(data.shape)

# center and scale the data
scaled_data = preprocessing.scale(data.T)

"""
NOTE: The preprocessing.scale() function expects the samples as rows and features as columns.
We use samples as columns in this example because that is often how genomic data is organized.
If you have other data, you can store it however is easiest for you. 
There's no requirement that samples be rows or columns, just be aware that if it is columns,
you'll need to transpose your data before analysis.
"""

# PCA
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

# calculate the percentage of variation that each principal component accounts for
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

# create labels for scree plot
labels = ['PC' + str(i) for i in range(1, len(per_var)+1)]

# plot the scree plot
plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

# plot the PCA data
pca_df = pd.DataFrame(pca_data, index=[*wt, *ko], columns=labels)

plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('My PCA Graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))

for sample in pca_df.index:
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))

plt.show()

# determine which genes had the biggest influence on PC1
# get the name of the top 10 measurements (genes) that contribute most to pc1
# first, get the loading scores
loading_scores = pd.Series(pca.components_[0], index=genes)

# now sort the loading scores based on their magnitude
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)

# get the names of the top 10 genes
top_10_genes = sorted_loading_scores[0:10].index.values

# print the gene names and their scores (and +/- sign)
print(loading_scores[top_10_genes])
