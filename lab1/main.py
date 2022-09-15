import math
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

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

def plot_housing():
    housing = pd.read_csv("housing.csv")
    # a
    num_of_districts = housing.shape[0]
    print(num_of_districts)
    # b
    mean(housing["median_house_value"])

    # c
    figure, axis = plt.subplots(2, 2)

    axis[0, 0].hist(housing["median_income"])
    axis[0, 0].set_title("median_income")

    axis[0, 1].hist(housing["housing_median_age"])
    axis[0, 1].set_title("housing_median_age ")

    axis[1, 0].hist(housing["median_house_value"])
    axis[1, 0].set_title("median_house_value")

    axis[1, 1].hist(num_of_districts)
    axis[1, 1].set_title("num_of_districts")
    plt.show()

if __name__ == '__main__':
    print("Min:")
    print(min(grades))
    print("")
    print("Max:")
    print(max(grades))
    print("")
    print("Range:")
    print(range(grades))
    print("")
    print("Mean:")
    print(mean(grades))
    print("")
    print("Variance:")
    print(variance(grades))
    print("")
    print("Standard deviation:")
    print(std_dev(grades))
    print("")
    print("Median:")
    print(median(grades))
    print("")
    print("Mean absolute deviation:")
    mean_absolute_deviation(grades)
    plot_housing()
