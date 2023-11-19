from sklearn.decomposition import PCA
import pickle
from helper import DataLoader, Helper, Metrics
import numpy as np
import os
import argparse
import pandas as pd
from scipy import stats


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
            print(train_var)
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
        df_X_test = df_X_test.sample(frac=1, random_state=1)
        X_test = df_X_test[cols_n[0]+cols_n[1]].to_numpy()
        
        sample_size = X_test.shape[0]//30

        test_var_list = np.zeros((30))
        for idx in range(30):
            if idx!=29:
                temp_X_test = X_test[idx*sample_size:(idx+1)*sample_size]
            else:
                temp_X_test = X_test[idx*sample_size:]
            temp_X_test_transform = pca.transform(temp_X_test)
            test_var = m.var_explained(temp_X_test, temp_X_test_transform@pca.components_+pca.mean_)
            test_var_list[idx] = test_var
            
            res = [train, train_var, test, test_var]
            print(res)
            final_report.append(res)



pd.DataFrame(final_report, columns=columns).to_csv("Results/Performance_at_95_TrainVE_30folds.csv")
