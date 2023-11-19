from sklearn.decomposition import PCA
from helper import DataLoader, Helper, Metrics
import pickle
import numpy as np
import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('data_name', metavar='N', type=str)

args = parser.parse_args()
data_name = args.data_name


d = DataLoader()

X, _, cols_n = d.delta_loader("Data/"+data_name+".csv", 68)
X = X[cols_n[0]+cols_n[1]]
X = X.to_numpy()
print(X.shape)

h = Helper()

for val in range(1, 137):
    pca = PCA(n_components=val, random_state=1)
    pca.fit(X)
    X_transform = pca.transform(X)
    m = Metrics(train_data=data_name, test_data="dump")
    train_var = m.var_explained(X, X_transform@pca.components_+pca.mean_)
    print(val, train_var)
    if(train_var)>=95:
        break


model_dir = "Results/"+str(val)+"_comp/"
try:
    os.makedirs(model_dir)
except:
    pass

### Below list is for non-interpretable keypoint movements for PCA AUs on DISFA_train
blue_kp_list = [None, None, [55,56,57,58,59], None, None, [56,57,58,65,66,67], [17,26], [17,26]]

outdir = model_dir+"pcaAUs_vis/"

try:
    os.makedirs(outdir)
except:
    pass

h.vis(X, pca.components_, pca.components_, outdir, [0, 68], X_transform, model_dir, 100, algo='pca', blue_kp_list=blue_kp_list)