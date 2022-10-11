import math
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import random
import csv

def load_data(filename):
    with open(filename, 'r', encoding='UTF-8') as f:
        csv_reader = csv.reader(f)
        df = pd.DataFrame(csv_reader)
        df = df.set_axis(df.iloc[0], axis=1, inplace=False)
        df = df.iloc[: , 1:]
        df = df.iloc[1: , :]
    return df

def mean(array):
    return sum(array) / len(array)

def mean_income(df, age):
    mean_income = []
    x = age
    for i in x:
        mean_income.append(mean(df[df["age"] == i]["2020"]))
    return np.array(mean_income)

def mse(true_val, pred_val):
    return np.square(np.subtract(true_val,pred_val)).mean()

def lin_reg_sk(df):
    lin_reg = LinearRegression()
    x = df["age"].to_numpy().reshape(-1, 1)
    y = mean_income(income_data, np.arange(16, 101))
    lin_reg.fit(x, y)
    return lin_reg

def grid_search(x, y, min_val, max_val):
    lin_reg_poly = LinearRegression()
    optimal_degree = 0
    mse_prev = math.inf
    for i in range(min_val, max_val+1):
        poly_features = PolynomialFeatures(degree=i, include_bias=False)
        x_poly = poly_features.fit_transform(x)
        lin_reg_poly.fit(x_poly, y)
        pred = lin_reg_poly.predict(x_poly)
        mse_prel = mse(y, pred)
        #print(mse_prel)
        if mse_prel < mse_prev:
            mse_prev = mse_prel
            optimal_degree = i
    return optimal_degree

class Kmeans:
    def __init__(self, K, N=10):
        self.K = K
        self.N = N
        self.data = None
        self.cluster_centers = pd.DataFrame()
        self.distance_df = pd.DataFrame(columns=["point", "closest_center", "distance"])
        self.old_c = []
        self.new_c = []
        self.silhouette_score = None
        
    def fit(self, df):
        self.data = df
        
        x = df.iloc[:, [0]]
        y = df.iloc[:, [1]]

        xlim = [np.array(x).min(), np.array(x).max()]
        ylim = [np.array(y).min(), np.array(y).max()]
        
        
        self.init_centroids(xlim, ylim)
        
        for i in range(self.N):
            self.euclidian_distance_fit()
            end_check = self.recenter()
            if end_check:
                return
        
    def init_centroids(self, xlim, ylim):
        column_names = self.data.columns.values
        for i in range(self.K):
            x = random.uniform(xlim[0], xlim[1])
            y = random.uniform(ylim[0], ylim[1])
            temp = pd.DataFrame([[x, y]], columns=column_names)
            self.cluster_centers = pd.concat([self.cluster_centers, temp], ignore_index=True)

    
    def euclidian_distance_fit(self, optinoal_data=[]):
        self.distance_df = pd.DataFrame(columns=["point", "closest_center", "distance"])
        for index1, row1 in self.data.iterrows():
            temp = []
            for index2, row2 in self.cluster_centers.iterrows():
                distance = np.linalg.norm(row2-row1)
                temp.append(distance)
            distance = min(temp)
            closest_center = min(range(len(temp)), key=temp.__getitem__)
            point = np.array([row1.iloc[0], row1.iloc[1]])
            temp_df = pd.DataFrame([[point, closest_center, distance]], columns=["point", "closest_center", "distance"])
            self.distance_df = pd.concat([self.distance_df, temp_df], ignore_index=True)
    
    def euclidian_distance_predict(self, df):
        return_df = pd.DataFrame(columns=["point", "closest_center", "distance"])
        for index1, row1 in df.iterrows():
            temp = []
            for index2, row2 in self.cluster_centers.iterrows():
                distance = np.linalg.norm(row2-row1)
                temp.append(distance)
            distance = min(temp)
            closest_center = min(range(len(temp)), key=temp.__getitem__)
            point = np.array([row1.iloc[0], row1.iloc[1]])
            temp_df = pd.DataFrame([[point, closest_center, distance]], columns=["point", "closest_center", "distance"])
            return_df = pd.concat([return_df, temp_df], ignore_index=True)
        
        return return_df
           
    
    def recenter(self):
        self.new_c = []
        counter = 0
        for i in range(self.K):
            closest_center = self.distance_df[self.distance_df["closest_center"] == i]
            size = closest_center["point"].size
            if size > 0:
                self.new_c.append(closest_center["point"].sum() / size)
            else:
                self.new_c.append(np.array([self.cluster_centers.iloc[i][0], self.cluster_centers.iloc[i][1]]))
            counter += 1
        self.cluster_centers = pd.DataFrame(np.array(self.new_c), columns=self.cluster_centers.columns.values)
        if len(self.new_c) == len(self.old_c):
            if np.sum(np.array(self.new_c) - np.array(self.old_c)) == 0:
                return True
        self.old_c = self.new_c
        return False
    
    def plot_clusters(self, title=None):
        for i in range(self.distance_df["closest_center"].max() + 1):
            points = self.distance_df[self.distance_df["closest_center"] == i]["point"]
            x = []
            y = []
            for i in points:
                x.append(i[0])
                y.append(i[1])
            plt.scatter(x, y)
        plt.scatter(self.cluster_centers.iloc[:,0], self.cluster_centers.iloc[:,1], marker='*', s=150, color='r', label="Centroids")
        plt.xlabel(self.data.columns[0])
        plt.ylabel(self.data.columns[1])
        if title:
            plt.title(title)
        plt.legend()
        plt.show()
        
    def plot_predictions(self, df, title=None):
        for i in range(self.distance_df["closest_center"].max() + 1):
            points = self.distance_df[self.distance_df["closest_center"] == i]["point"]
            x = []
            y = []
            for i in points:
                x.append(i[0])
                y.append(i[1])
            plt.scatter(x, y)   
        plt.scatter(self.cluster_centers.iloc[:,0], self.cluster_centers.iloc[:,1], marker='*', s=150, color='r', label="Centroids")
        
        for index, row in  self.cluster_centers.iterrows():
            plt.annotate(f"C{index}", xy=(np.array(row)[0], np.array(row)[1]))
        
        plt.xlabel(self.data.columns[0])
        plt.ylabel(self.data.columns[1])
        
        x_pred = []
        y_pred = []
        for i in df["point"]:
            x_pred.append(i[0])
            y_pred.append(i[1])
        
        plt.scatter(x_pred, y_pred, marker='^', label="Predicted Values", color='r', s=100)
        
        for index, row in  df.iterrows():
            plt.annotate(f"C{row['closest_center']}", xy=(np.array(row["point"])[0], np.array(row["point"])[1]))
        
        if title:
            plt.title(title)
        plt.legend()
        plt.show()
    
    def avg_intra_cluster_distance(self, point, cluster):
        cluster_points = self.distance_df[self.distance_df["closest_center"] == int(cluster)]
        dist_list = []
        for i in np.array(cluster_points["point"]):
            dist_list.append(np.linalg.norm(i-point))
        return sum(dist_list) / len(dist_list)

    def avg_inter_cluster_distance(self, point, cluster):
        dist_dict = {}
        centers = self.cluster_centers.copy(deep=False)
        centers.drop(labels=cluster)

        for index, row in centers.iterrows():
            dist_dict[index] = np.linalg.norm(row-point)
        del dist_dict[cluster]
        closest_cluster = min(dist_dict, key=dist_dict.get)
        cluster_points = self.distance_df[self.distance_df["closest_center"] == int(closest_cluster)]
        dist_list = []

        if len(cluster_points["point"]) > 0:
            
            for i in np.array(cluster_points["point"]):
                dist_list.append(np.linalg.norm(i-point))
                return sum(dist_list) / len(dist_list)
        else:
            return np.linalg.norm(centers.iloc[closest_cluster] - point)
            
    
    def silhouette_coef(self):
        silhouette_coefs = []
        points = self.distance_df
        for index, row in points.iterrows():
            a = self.avg_intra_cluster_distance(row["point"], row["closest_center"])
            b = self.avg_inter_cluster_distance(row["point"], row["closest_center"])
            if a == 0 or b == 0:
                S = 0
            else:
                S = (b - a) / max([a, b])
            silhouette_coefs.append(S)
        self.distance_df["silhouette_coef"] = silhouette_coefs
        self.silhouette_score = self.distance_df["silhouette_coef"].mean()

    def predict(self, df):
        distances = self.euclidian_distance_predict(df)
        return distances     


def optimize_Kmeans(data, min_k, max_k):
    if min_k <= 1:
        print("min_k must be equal to or larger than 2")
        return None
    silhouette_scores = []
    all_ks = []
    old_sil_best = -1
    old_sil_worst = 1
    return_best_k = None
    return_worst_k = None
    for i in range(min_k, max_k):
        k1 = Kmeans(K=i)
        k1.fit(rent_inc)
        k1.silhouette_coef()
        silhouette_scores.append(k1.silhouette_score)
        all_ks.append(k1)
        if k1.silhouette_score > old_sil_best:
            old_sil_best = k1.silhouette_score
            return_best_k = k1
        if k1.silhouette_score < old_sil_worst:
            old_sil_worst = k1.silhouette_score
            return_worst_k = k1
    return silhouette_scores, return_best_k, return_worst_k, all_ks

def plot_silhouette(k_arr):
    for i, k in enumerate(k_arr):
        y_lower = y_upper = 0
        #if i > 0:
        for j in range(0, k.distance_df["closest_center"].max() + 1):
            cluster_silhouette_vals = k.distance_df[k.distance_df["closest_center"] == j]["silhouette_coef"].sort_values()
            #print(j)
            #print(cluster_silhouette_vals)
            #print("----")
            y_upper += len(cluster_silhouette_vals)
            
            plt.barh(range(y_lower,y_upper),cluster_silhouette_vals,height =1) 
            
            plt.text(-0.03,(y_lower+y_upper)/2,str(j))
            y_lower += len(cluster_silhouette_vals)
    
        plt.yticks([])
        plt.xlim([-0.1, 1])
        plt.xlabel('Silhouette coefficient values')
        plt.ylabel('Cluster labels')
        plt.title('Silhouette plot for the various clusters');
        plt.tight_layout()
        plt.show()
    

if __name__ == '__main__':
    #TASK 1
    income_data = load_data("inc_utf.csv")
    income_data["age"] = income_data["age"].str.extract('(\d+)').astype("int")
    income_data["2020"] = pd.to_numeric(income_data["2020"])
    x = np.arange(16, 101).reshape(-1, 1)
    y = mean_income(income_data, np.arange(16, 101))
    
    lin_reg_income = LinearRegression()
    lin_reg_income.fit(x, y)

    #Predict (35, 80)
    pred1 = lin_reg_income.predict(np.array([[35], [80]])) #(287.70961212, 269.45876341) - lab2 [287.7096121239193, 269.4587634123417]
    
    #Linear prediction and MSE
    y_pred = lin_reg_income.predict(x.reshape(-1, 1))
    mse_income = mean_squared_error(y.reshape(-1, 1), y_pred) #27890.405823634588 - lab2 #27890.405823634545
    
    #TASK 2
    lin_reg_poly = LinearRegression()
    
    #Finding optimal degree
    optimal_degree = grid_search(x, y, 2, 10)
    
    #Using the optimal degree to fit the model and get the prediciton
    poly_features = PolynomialFeatures(degree=optimal_degree, include_bias=False)
    x_poly = poly_features.fit_transform(x)
    lin_reg_poly.fit(x_poly, y)
    pred2 = lin_reg_poly.predict(x_poly)

    #Plotting the income data, polynomial and linear prediction 
    plt.scatter(x, y)
    plt.plot(x, pred2, label='Polynomial Regression', color='g')
    plt.plot(x, y_pred, label='Linear Regression', color='r')
    plt.legend(loc='lower right')
    plt.show()
    
    #TASK 3
    inc_vs_rent = load_data('inc_vs_rent.csv')
    rent_inc = inc_vs_rent.filter(['Annual rent sqm', 'Avg yearly inc KSEK'], axis=1)
    rent_inc["Annual rent sqm"] = pd.to_numeric(rent_inc["Annual rent sqm"])
    rent_inc["Avg yearly inc KSEK"] = pd.to_numeric(rent_inc["Avg yearly inc KSEK"])
    
    k1 = Kmeans(K=2)
    k1.fit(rent_inc)
    k1.plot_clusters("My model")
    
    #TASK 4
    opt_min = 2
    opt_max = 7
    
    y, k, k_bad, k_all = optimize_Kmeans(rent_inc, opt_min, opt_max)
    
    plot_silhouette(k_all)
    
    x = np.arange(opt_min, opt_max)
    plt.plot(x, y, '-bo')
    plt.ylabel("Silhouette score")
    plt.show()
    
    k.plot_clusters(title="Best model")
    k_bad.plot_clusters(title="Worst model")
    
    unnamed_regions =  pd.DataFrame([[1010, 320.12], [1258, 320], [980, 292.4]],
                                    columns = ["Annual rent sqm", "Avg yearly inc KSEK"])
    k.plot_predictions(k.predict(unnamed_regions), title="Prediction")
    
    #Additional excer