# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 18:45:50 2019

@author: LAURI
"""

import numpy as np
import seaborn as sb
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def display_figure(name,m):
    X[name]=m.labels_
    sb.catplot(x='X',y='Y',data=X, hue=name)
    plt.title(name)


X, labels = make_blobs(n_samples=50, n_features=2)
X = pd.DataFrame(X,columns=['X','Y'])


from sklearn.cluster import KMeans
clustering_kmeans = KMeans(n_clusters=2).fit(X)

from sklearn.cluster import MeanShift
clustering_meanshift = MeanShift().fit(X)

from sklearn.cluster import DBSCAN
clustering_dbscan = DBSCAN(eps=3, min_samples=2).fit(X)


display_figure('KMeans',clustering_kmeans)
display_figure('MeanShift',clustering_meanshift)
display_figure('DBScan',clustering_dbscan)
