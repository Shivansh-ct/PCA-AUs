from sklearn.decomposition import PCA
from helper import DataLoader, Metrics
import pickle
import numpy as np
import os
import argparse
import pandas as pd


train_data = ["DISFA_train", "BP4D_train", "CK+"]
test_data = ["DISFA_full", "CK+", "BP4D_full"]

d = DataLoader()

final_report = []
columns = ["Train_Name","Test_Name", "Components", "Var_Type", "Var_Explained"]


model_list = []

for train in train_data:
    df_X_train, _, cols_n = d.delta_loader("Data/"+train+".csv")
    X_train = df_X_train[cols_n[0]+cols_n[1]].to_numpy()

    pca = PCA(n_components=X_train.shape[1], random_state=1)
    pca.fit(X_train)
    if "DISFA" in train:
        pcaAUs = pca.components_
        pd.DataFrame(pcaAUs).to_csv("Data/delta_pcaAUs.csv")

    m = Metrics(train_data=train, test_data="dump")

    for i in range(1,X_train.shape[1]+1):
        temp_components = pca.components_[:i,:]
        X_train_transform = (X_train-pca.mean_)@temp_components.T
        train_var = m.var_explained(X_train, X_train_transform@temp_components+pca.mean_)
        res = [train,train,i,'train_var',train_var]
        final_report.append(res)


    if "DISFA" in train:
        temp_test_name = "DISFA"
    elif "BP4D" in train:
        temp_test_name = "BP4D"
    else:
        temp_test_name = "CK+"


    for test in test_data:
        if temp_test_name in test:
            continue
        m = Metrics(train, test)
        res = []
        df_X_test, _, cols_n = d.delta_loader("Data/"+test+".csv")
        
        X_test = df_X_test[cols_n[0]+cols_n[1]].to_numpy()

        for i in range(1,X_train.shape[1]+1):
            print(train, test, i)
            temp_components = pca.components_[:i,:]
            X_test_transform = (X_test-pca.mean_)@temp_components.T
            test_var = m.var_explained(X_test, X_test_transform@temp_components+pca.mean_)
            res = [train,test,i,'test_var',test_var]
            final_report.append(res)

print("saving the file...")
pd.DataFrame(final_report, columns=columns).to_csv("Results/all_components_performance.csv")


