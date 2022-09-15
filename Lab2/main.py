import math
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

#Lab 1
def min(array):
    temp = math.inf
    for i in array:
        if i < temp:
            temp = i
    return temp

def max(array):
    temp = math.inf * -1
    for i in array:
        if i > temp:
            temp = i
    return temp

def range(array):
    return max(array) - min(array)

def mean(array):
    return sum(array) / len(array)

def variance(array):
    # calculate mean
    array_mean = mean(array)

    # deviation from mean
    deviation_from_mean = []
    for i in array:
        deviation_from_mean.append(i - array_mean)

    # squeare deviations, sum and return variance
    deviations_squared = [i ** 2 for i in deviation_from_mean]
    sum_deviations_squared = sum(deviations_squared)
    return sum_deviations_squared / (len(array) - 1)

def std_dev(array):
    return math.sqrt(variance(array))

def median(array):
    array_sorted = sorted(array)
    if len(array) % 2 == 0:
        return (array_sorted[(int(len(array) / 2) - 1)] + array_sorted[int((len(array) / 2) + 1)]) / 2
    else:
        return array_sorted[math.ceil(len(array) / 2)]

def mean_absolute_deviation(array):
    array_median = median(array)
    deviation_from_medeian_abs = []
    for i in array:
        deviation_from_medeian_abs.append(abs(i - array_median))
    return mean(deviation_from_medeian_abs)

#Lab2
def load_data(filename):
    return pd.read_csv(filename)

def mean_pop_post_sec_norrbotten(df):
    norrbotten = df[df["region"] == "25 Norrbotten county"].iloc[:, -5:].astype(int)
    return sum(norrbotten.iloc[0])

def std_dev_pop_post_sec_norrbotten(df):
    norrbotten = df[df["region"] == "25 Norrbotten county"].iloc[:, -5:].astype(int)
    return std_dev(norrbotten.iloc[0])

def sum_pop_all_edu_levels(df, regions):
    return_df = pd.DataFrame()
    for region in regions:
        current_region = df[df["region"] == region].iloc[:, -5:].astype(int)
        temp_df = pd.DataFrame([[region,
                                 sum(current_region["2016"]),
                                 sum(current_region["2017"]),
                                 sum(current_region["2018"]),
                                 sum(current_region["2019"]),
                                 sum(current_region["2020"])]],
                               columns=["region", "2016", "2017", "2018", "2019", "2020"])
        return_df = pd.concat([return_df, temp_df])
    return return_df

def mean_population_all_regions(df):
    return_dict = {}
    for index, row in df.iterrows():
        return_dict[row["region"]] = mean(row[1:,])
    return return_dict

def histogram_2020(df):
    x = np.arange(0, 21)
    plt.bar(x, df["2020"], label="2020")
    plt.xticks(x, df["region"], rotation='vertical')
    plt.tight_layout()
    plt.legend()
    plt.show()

def sum_men_women(df):
    return_df = pd.DataFrame()

    for index, row in df.iterrows():
        if index == 0:
            last_row = row
        else:
            if (last_row["region"] == row["region"]) and (last_row["level of education"] == row["level of education"]):
                temp_df = pd.DataFrame([[row["region"],
                                         row["level of education"],
                                         (row["2016"] + last_row["2016"]) / 2,
                                         (row["2017"] + last_row["2017"]) / 2,
                                         (row["2018"] + last_row["2018"]) / 2,
                                         (row["2019"] + last_row["2019"]) / 2,
                                         (row["2020"] + last_row["2020"]) / 2]],
                                       columns=["region", "level of education", "2016", "2017", "2018", "2019", "2020"])
                return_df = pd.concat([return_df, temp_df])
                last_row = row
            else:
                last_row = row
    return return_df

def scatter_merged_data(df):
    #data processing
    no_info = df[df["level of education"] == "no information about level of educational attainment"]
    post_secondary = df[df["level of education"] == "post secondary education"]
    primary = df[df["level of education"] == "primary and lower secondary education"]
    upper_secondary = df[df["level of education"] == "upper secondary education"]
    
    #plotting
    plt.scatter(no_info["2020_x"], no_info["2020_y"], c="blue", label="no information about level of educational attainment")
    plt.scatter(post_secondary["2020_x"], post_secondary["2020_y"], c="orange", label="post secondary education")
    plt.scatter(primary["2020_x"], primary["2020_y"], c="green", label="primary and lower secondary education")
    plt.scatter(upper_secondary["2020_x"], upper_secondary["2020_y"], c="red", label="upper secondary education")
    
    #labeling
    plt.legend()
    plt.xlabel("Population amount")
    plt.ylabel("Avg yearly income")
    plt.tight_layout()
    
    plt.show()
    
def lin_reg_income_pop(df):
    lin_reg = LinearRegression()
    x = df["2020_x"].to_numpy().reshape(-1, 1)
    #x = x.reshape(-1, 1)
    y = df["2020_y"].to_numpy().reshape(-1, 1)
    
    #print(x)
    lin_reg.fit(x, y)
    
    x_new = np.array([[0],[850000]])
    y_predict = lin_reg.predict(x_new)
    
    plt.plot(x_new, y_predict, "r-")
    plt.scatter(x, y)
    
    plt.xlabel("Population amount")
    plt.ylabel("Avg yearly income")
    plt.tight_layout()
    plt.show()
    
    return lin_reg

def mse(df, lin_reg):
    y_true = df["2020_y"].to_numpy().reshape(-1, 1)
    y_pred = lin_reg.predict(y_true)
    
    return np.square(np.subtract(y_true,y_pred)).mean()

def lin_reg_post_secondary(df):
    post_secondary = df[df["level of education"] == "post secondary education"]
    return lin_reg_income_pop(post_secondary)

if __name__ == '__main__':    
    #Task 1
    data = load_data("pop_year_trim.csv")

    #Task 2
    post_secondary = data[data["level of education"] == "post secondary education"]
    region_names = post_secondary["region"]
    post_secondary = post_secondary.drop("level of education", axis=1)
    sum_population_edu_levels = sum_pop_all_edu_levels(data, region_names)
    mean_population_all_regions = mean_population_all_regions(sum_population_edu_levels)
    histogram_2020(sum_population_edu_levels)

    #Task 3
    data2 = load_data("avg_inc_2.csv")
    
    #Task 4
    sum_men_women = sum_men_women(data2)
    merged_data = pd.merge(data, sum_men_women, on=["region", "level of education"]) #population=x, income=y
    scatter_merged_data(merged_data)

    #Task 5
    lin_reg = lin_reg_income_pop(merged_data)
    predicted_values = lin_reg.predict(np.array([[20000], [80000]])) #(229.87, 246.41)
    mse_all = mse(merged_data, lin_reg) #9603.51
    
    lin_reg_post_secondary = lin_reg_post_secondary(merged_data)
    predicted_values_post_secondary = lin_reg_post_secondary.predict(np.array([[20000], [80000]])) #(379.22, 380.92)
    mse_post_secondary = mse(merged_data[merged_data["level of education"] == "post secondary education"], lin_reg_post_secondary) #328.98