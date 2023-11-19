import os
import cv2
import matplotlib
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from pychubby.detect import LandmarkFace
from pychubby.actions import Chubbify, Multiple, Pipeline, Smile, AbsoluteMove, Action, Lambda
from skimage.transform import SimilarityTransform, AffineTransform


### replace the below two lines with your image and keypoints to project your PCA components
face_kp_n = np.loadtxt("../neutral_faceImage_68keypoints.txt", delimiter=",")     
inp_path_n = "../neutral_faceImage.jpg"


class DataLoader:

    def __init__(self):
        self.data = None
        self.delta_data = None
        self.kp_columns = None
        self.kp_n_columns = None
        

    def delta_loader(self, input_file_path=None, df_data=None, n_kp=68, vid=None):
        # input_file_path and df_delta can't be None at the same time
        if input_file_path is not None:
            df_delta = pd.read_csv(input_file_path, index_col=0, dtype={'Subject_id':str, 'Video_id':str})        
        else:
            df_delta = df_data.copy()

        if vid is not None:
            df_delta = df_delta[df_delta["Video_id"]==vid]
        delta_column_list = []
        name_list = ['x', 'y']
        name = ["", "_Normalized"]

        for name2 in name:
            for i, name1 in product(range(n_kp), name_list):
                df_delta["Apex_"+name1+str(i)+name2] = df_delta["Apex_"+name1+str(i)+name2] -df_delta["Offset_"+name1+str(i)+name2]
                df_delta.drop(columns=["Offset_"+name1+str(i)+name2], inplace=True)
                col = "delta_"+name1+str(i)+name2
                delta_column_list.append(col)
                df_delta.rename(columns={"Apex_"+name1+str(i)+name2:col},  inplace=True)
                
                
        counter = 0

        x_column = []
        y_column = []
        x_n_column = []
        y_n_column = []

        for i in delta_column_list:

            if counter <2*n_kp:
                if counter%2==0:
                    x_column.append(i)
                else:
                    y_column.append(i)
            else:
                if counter%2==0:
                    x_n_column.append(i)
                else:
                    y_n_column.append(i)  

            counter+=1
        
        self.delta_data, self.kp_columns, self.kp_n_columns = df_delta, [x_column, y_column], [x_n_column, y_n_column]
            
        return df_delta, [x_column, y_column], [x_n_column, y_n_column]     




class Metrics:

    def __init__(self, train_data=None, test_data=None):
        self.train_data=train_data
        self.test_data = test_data
        self.var_explained_ = None
        self.avg_var_explained_ = None
        self.avg_row_nonzero_ = None

        
    def var_explained(self, X, X_approx):

        if "DISFA" in self.train_data or "DISFA" in self.test_data:
            X[:, 60], X[:, 64] = 0, 0
            X[:, 60+68], X[:, 64+68] = 0, 0
            X_approx[:, 60], X_approx[:, 64] = 0, 0
            X_approx[:, 60+68], X_approx[:, 64+68] = 0, 0

        if "BP4D" in self.train_data or "BP4D" in self.test_data:
            X[:, 60], X[:, 64], X[:, :17] = 0, 0, 0
            X[:, 60+68], X[:, 64+68], X[:, 68:(68+17)] = 0, 0, 0
            X_approx[:, 60], X_approx[:, 64], X_approx[:, :17] = 0, 0, 0
            X_approx[:, 60+68], X_approx[:, 64+68], X_approx[:, 68:(68+17)] = 0, 0, 0

        # X and X_approx need to be numpy array
        ve = 100 - 100*(np.linalg.norm(X - X_approx)**2)/(np.linalg.norm(X)**2)
        self.var_explained_ = ve    
        return ve



class Transform:
    def __init__(self, inp_path=None, matrix=None):
        self.path = inp_path
        self.matrix = matrix
        

    def pychubby(self, inp_path, face_kp, K, xdiff, ydiff, out_path, std_dev=None, blue_kp=None):

        class CustomAction(Action):
            def __init__(self, scale=0.3):
                self.scale = scale
            def perform(self, lf):
                a_l = AbsoluteMove(x_shifts=xdiff_final,y_shifts=ydiff_final)
                return a_l.perform(lf)

        def get_all_au():
            img = cv2.imread(inp_path)
            lf = LandmarkFace.estimate(img)
            lf.points = face_kp
            xdiff_final = dict(enumerate(xdiff*K,start=0))
            ydiff_final = dict(enumerate(ydiff*K,start=0))
            return xdiff_final,ydiff_final,lf

        path = inp_path
        xdiff_final,ydiff_final,lf = get_all_au()
        a_all = CustomAction()
        new_lf, _ = a_all.perform(lf)
    
        thickness = 3
        img = new_lf.img.copy()

        # print(blue_kp)
        for i in range(68):
            if (blue_kp is not None) and (i in blue_kp):
                # print("utilizing the blue_kp")
                start_point = (int(face_kp[i, 0]), int(face_kp[i,1]))
                end_point = (int(face_kp[i, 0]+xdiff_final[i]),  int(face_kp[i, 1]+ydiff_final[i]))
                dist = np.linalg.norm(np.array(start_point) - np.array(end_point)) + 1e-15
                tiplength = 5/dist
                img = cv2.circle(img, end_point, 7, (255, 255, 0), -1)
                img = cv2.arrowedLine(img, start_point, end_point, (0, 0, 0), thickness, tipLength=tiplength)
                img = cv2.circle(img, start_point, 7, (0, 0, 255), -1)

            else:
                start_point = (int(face_kp[i, 0]), int(face_kp[i,1]))
                end_point = (int(face_kp[i, 0]+xdiff_final[i]),  int(face_kp[i, 1]+ydiff_final[i]))
                dist = np.linalg.norm(np.array(start_point) - np.array(end_point)) + 1e-15
                tiplength = 5/dist
                img = cv2.circle(img, end_point, 7, (0, 255, 0), -1)
                img = cv2.arrowedLine(img, start_point, end_point, (0, 0, 0), thickness, tipLength=tiplength)
                img = cv2.circle(img, start_point, 7, (0, 0, 255), -1)

        new_lf.img = img
        cv2.imwrite(out_path, new_lf.img)
        return new_lf.img
    
  

    
class Helper:
    def __init__(self):
        pass

    def draw_keypoints(self, kps, save_path=None, rad=3):
        maxx = np.amax(kps[:, 0])
        maxy = np.amax(kps[:, 1])
        img = np.zeros((int(1.25*maxy), int(1.25*maxx)))
        for x, y in zip(kps[:, 0], kps[:, 1]):
            img = cv2.circle(img, (int(x), int(y)), rad, (255, 255, 255), -1) 

        if save_path is not None:
            cv2.imwrite(save_path, img)

    def var_explained(self, X, X_approx):
        return 100 - 100*(np.linalg.norm(X - X_approx)**2)/(np.linalg.norm(X)**2)

    def var_and_cumvar(self, X, loadings, X_transform=None, outdir=None, algo=None, train_data=None, test_data=None, part=None, train_mean=None):
        
        var_arr = np.zeros((loadings.shape[0],2))
        var_arr[:,0] = np.arange(0, loadings.shape[0])

        h = Helper()
        m = Metrics(train_data, test_data)

        for i in range(loadings.shape[0]):
            comp = loadings[i]
            comp = comp.reshape(1, loadings.shape[1])
            if X_transform is None:
                X_approx = (X@np.linalg.pinv(comp))@comp
            else:
                X_approx = (X_transform[:, i].reshape(X_transform.shape[0], 1))@(loadings[i, :].reshape(1, loadings.shape[1]))
            if algo=="pca" or algo=="spca":
                X_approx += np.mean(X, axis=0)
            
            if (train_data is not None) or (test_data is not None):
                var = m.var_explained(X, X_approx)
            else:
                var = 100*(1 - (np.linalg.norm(X - X_approx)**2)/(np.linalg.norm(X)**2))
            
            var_arr[i, 1] = var
        sorted_var_arr = h.sort(var_arr.copy(), 0, 1)

        cum_var_arr = np.zeros((loadings.shape[0],3))
        cum_var_arr[:,0] = sorted_var_arr[:,0]
        
        cum_comp_nonzero = np.zeros((loadings.shape[0], 1))
        for i in range(cum_var_arr.shape[0]):
            index = int(cum_var_arr[i,0])
            if X_transform is None:

                if i==0:
                    comp = loadings[index]
                    comp = comp.reshape(1, loadings.shape[1])
                else:
                    comp = np.concatenate((comp, loadings[index].reshape(1, loadings.shape[1])), axis=0)
                X_approx = (X@np.linalg.pinv(comp))@comp

            else:
                if i==0:
                    X_transformed_sorted = X_transform[:, index].reshape(X_transform.shape[0], 1)
                    comp = loadings[index]
                    comp = comp.reshape(1, loadings.shape[1])
                else:
                    X_transformed_sorted = np.concatenate((X_transformed_sorted, X_transform[:, index].reshape(X_transform.shape[0], 1)), axis=1)
                    comp = np.concatenate((comp, loadings[index].reshape(1, loadings.shape[1])), axis=0)
                X_approx = X_transformed_sorted@comp

            cum_comp_nonzero[i, 0] = np.count_nonzero(comp)
            if (algo=="pca") and (train_mean is not None):
                X_approx+=train_mean
            
            if (train_data is not None) or (test_data is not None):
                var = m.var_explained(X, X_approx)
            else:   
                var = 100*(1 - (np.linalg.norm(X - X_approx)**2)/(np.linalg.norm(X)**2))
            cum_var_arr[i, 1] = var
            cum_var_arr[i, 2] = m.avg_row_nonzero(X_transformed_sorted)
    
        if outdir is not None:
            np.set_printoptions(suppress=True)
            np.savetxt(outdir+"var_arr.txt", var_arr, delimiter=' , ', fmt="%f")
            np.savetxt(outdir+"sorted_var_arr.txt", sorted_var_arr, delimiter=' , ', fmt="%f")
            np.savetxt(outdir+"cum_var_arr.txt", cum_var_arr, delimiter=' , ', fmt="%f")

        if outdir is not None:
            df = pd.DataFrame(np.concatenate((sorted_var_arr, cum_var_arr[:, 1].reshape(cum_var_arr.shape[0], -1), cum_var_arr[:, 2].reshape(cum_var_arr.shape[0], -1)), axis=1), columns=['Components', 'Variance', 'Cum Variance', 'AC'])
            df.to_csv(outdir + "var_list.csv")
            df = pd.DataFrame(np.concatenate((cum_comp_nonzero, cum_var_arr[:, 1].reshape(cum_var_arr.shape[0], -1)), axis=1), columns=['Non Zeros in V', 'Cum Variance'])
            df.to_csv(outdir + "var_list_nonzero.csv")    
     
        return var_arr, sorted_var_arr, cum_var_arr





    def vis(self, X, loadings_var, loadings_vis, outdir, face_part, X_transform=None, var_dir=None, scale=100, algo=None, data_name=None ,blue_kp_list=None):
        a = face_part[0]
        b = face_part[1]
        trans = Transform()
        h = Helper()

        try:
            var_arr, _, _ = h.var_and_cumvar(X, loadings_var, X_transform, var_dir, algo, data_name)
        except Exception as e:
            var_arr = "NA"

        for i in range(loadings_vis.shape[0]):
            if np.sum(loadings_vis[i])!= 0:
                loadings_vis[i] = loadings_vis[i]/np.linalg.norm(loadings_vis[i])

        for i in range(0, loadings_vis.shape[0]):

            comp = np.zeros(136)
            comp[a:b] = loadings_vis[i, :(b-a)]
            comp[(a+68):(b+68)] = loadings_vis[i, (b-a):]
            comp = comp.reshape(1, 136)

            if var_arr!="NA":
                if blue_kp_list is not None:
                    trans.pychubby(inp_path_n, face_kp_n, scale, comp[0, :68], comp[0, 68:], outdir+"{:.2f}".format(var_arr[i,1])+"_"+str(i)+".jpg", blue_kp=blue_kp_list[i])
                
                else:
                    trans.pychubby(inp_path_n, face_kp_n, scale, comp[0, :68], comp[0, 68:], outdir+"{:.2f}".format(var_arr[i,1])+"_"+str(i)+".jpg")

            else:
                if blue_kp_list is not None:
                    trans.pychubby(inp_path_n, face_kp_n, scale, comp[0, :68], comp[0, 68:], outdir+str(i)+".jpg", blue_kp=blue_kp_list[i])
                
                else:
                    trans.pychubby(inp_path_n, face_kp_n, scale, comp[0, :68], comp[0, 68:], outdir+str(i)+".jpg")