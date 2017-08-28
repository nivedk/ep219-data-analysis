import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#This method uses the pandas method read_csv to extract data from a .txt file, which is then converted into a numpy array for ease of manipulation which is returned
def readData():
    #reads the data from .txt file and .values converts it to a numpy ndarray
    return pd.read_csv("facebookdata.csv",delim_whitespace=True, header=None).values
print (readData())
