import math
from matplotlib import pyplot as plt
import pandas as pd
#import numpy as np

grades = [8, 6, 1, 7, 8, 9, 8, 7, 10, 7, 6, 9, 7]

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
    return max(array) - min(array) #Should i return absolute value?

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



def plot_housing():
    housing = pd.read_csv("housing.csv")
    # a
    num_of_districts = housing.shape[0]
    print("Number of districts")
    print(num_of_districts)
    print("")
    # b
    print("Mean house values:")
    print(mean(housing["median_house_value"]))
    print("")

    # c
    figure, axis = plt.subplots(2, 2)

    axis[0, 0].hist(housing["median_income"])
    axis[0, 0].set_title("median_income")

    axis[0, 1].hist(housing["housing_median_age"])
    axis[0, 1].set_title("housing_median_age ")

    axis[1, 0].hist(housing["median_house_value"])
    axis[1, 0].set_title("median_house_value")

    axis[1, 1].hist(housing["households"])
    axis[1, 1].set_title("households")
    
    plt.suptitle("All Regions")
    plt.tight_layout()
    plt.show()
    
    #f
    ocean_prox = housing["ocean_proximity"].unique()
    print("Mean per district:")
    for i in ocean_prox:
        per_region = housing[housing["ocean_proximity"] == i]
        print("")
        print(i)
        print(mean(per_region["median_income"]))
        
        figure, axis = plt.subplots(2, 2)

        axis[0, 0].hist(per_region["median_income"])
        axis[0, 0].set_title("median_income")

        axis[0, 1].hist(per_region["housing_median_age"])
        axis[0, 1].set_title("housing_median_age ")

        axis[1, 0].hist(per_region["median_house_value"])
        axis[1, 0].set_title("median_house_value")

        axis[1, 1].hist(per_region["households"])
        axis[1, 1].set_title("households")
        
        plt.suptitle(i)
        plt.tight_layout()
        plt.show()
    return housing

if __name__ == '__main__':
    print("Min:") #The minimum value of a dataset
    print(min(grades))
    print("")
    print("Max:")
    print(max(grades)) # The maximum value of a dataset
    print("")
    print("Range:") 
    print(range(grades)) #Show how "wide" a dataset is
    print("")
    print("Mean:")
    print(mean(grades)) #Gives one number for the avarage of the dataset
    print("")
    print("Variance:") #Measueres variability
    print(variance(grades))
    print("")
    print("Standard deviation:") #Also measures variability, but is in the same unit of measurement as the original value
    print(std_dev(grades))
    print("")
    print("Median:") #The middle value of a dataset, useful if the data is skewed(for example salary in a country, mean and median will differ quite a bit)
    print(median(grades))
    print("")
    print("Mean absolute deviation:") #The avarage distance between all datapoints in a dataset and it's mean. Also a measure of variability
    print(mean_absolute_deviation(grades))
    print("")
    plt.hist(grades)
    plt.show()
    mean_absolute_deviation(grades)
    housing = plot_housing()
