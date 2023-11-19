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
columns = ["train_data", "train var", "test_data", "test var"]




for train in train_data:
    df_X_train, _, cols_n = d.delta_loader("Data/"+train+".csv")
    X_train = df_X_train[cols_n[0]+cols_n[1]].to_numpy()

    m = Metrics(train_data=train, test_data="dump")
    for n_comp in range(1, 137):
        pca = PCA(n_components=n_comp, random_state=1)
        pca.fit(X_train)
        X_train_transform = pca.transform(X_train)
        train_var = m.var_explained(X_train, X_train_transform@pca.components_+pca.mean_)
        if train_var>=95:
            break

    print("Train var is : ", train_var)

    for test in test_data:
        if "DISFA" in train and "DISFA" in test:
            continue
        if "BP4D" in train and "BP4D" in test:
            continue
        if "CK+" in train and "CK+" in test:
            continue
        m = Metrics(train, test)
        res = []
        df_X_test, _, cols_n = d.delta_loader("Data/"+test+".csv")
        
        X_test = df_X_test[cols_n[0]+cols_n[1]].to_numpy()
        X_test_transform = pca.transform(X_test)
        test_var = m.var_explained(X_test, X_test_transform@pca.components_+pca.mean_)
        
        train_comps = test_comps = pca.components_.shape[0]
        avg_train_comps = m.avg_row_nonzero(X_train_transform)
        avg_test_comps = m.avg_row_nonzero(X_test_transform)
        res = [train, "{:.5f}".format(train_var), test, "{:.5f}".format(test_var)]
        final_report.append(res)
        print(res)

pd.DataFrame(final_report, columns=columns).to_csv("Results/Performance_at_95_TrainVE.csv")