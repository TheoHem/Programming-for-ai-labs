import math
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def load_data(filename):
    return pd.read_csv(filename)

if __name__ == '__main__':
    pop_year = load_data("pop_year_trim.csv")
    avg_inc= load_data("avg_inc_2.csv")
