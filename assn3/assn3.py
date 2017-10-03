'''
Course: EP219 Fall 2017
Course instructor: Vikram Rentala
Assignment number: 3

Description: This program reads the 10,000 anonymous facebook user data (containing number of friends, number of posts, number of likes) collected over a period of 10 years (provided by the instructor) and calculates the covariance matrices and the correlation coefficients for each combination of data pairs. We then look at these relults and try to explain why they have taken the specific values that they have. 

Description of functions:
numpy.ndarray readData(): This method uses the pandas method read_csv to extract data from a .csv file, which is then converted into a numpy array for ease of manipulation which is returned
main : In this code, most of the processing happens in the main function

Version: 1.0

Requirements:
Python 2.7 with the following packages installed
    -numpy
    -pandas
    -matplotlib

How to run the code:
Make sure, the facebookdata.csv file is in the same direct folder as this file (assn3.py), and just run it on terminal/command prompt or any other IDE/program
'''
#pandas is used for reading from the csv file
import pandas as pd
#numpy is used for eaesy and efficient array operations
import numpy as np
#matplotlib is used for generating different types of plots
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rc

#This method uses the pandas method read_csv to extract data from a .csv file, which is then converted into a numpy array for ease of manipulation which is returned
def readData():
    #reads the data from .txt file and .values converts it to a numpy ndarray
    #Column 1(0) has friends, Column 2(1) has posts, Column 3(2) has likes
    return pd.read_csv("facebookdata.csv", header=None).values

if __name__ == "__main__":
    data = readData()
    data = (data[1:,:])
    data = data.astype(np.float)


    print("\nThe covariance matrix for friends vs posts\n")

    #np.cov is a numpy method that calculates the covariance matrix of two random variables given two N-dimensional samples
    #print(np.cov(data[:,0], data[:,1], data[:,2]))

    print("\nThe covariance matrix for posts vs likes\n")
    print(np.cov(data[:,1], data[:,2]))

    print("\nThe covariance matrix for likes vs friends\n")
    print(np.cov(data[:,2], data[:,0]))

    print("\n=====================================================================================\n")

    #Correlation coefficients to be found are (ff), (fp), (fl), (pp), (pl), (ll)
    #Correlation coefficient of two random variables is defined as their covariance divided by their respective standard deviations
    #np.std is a numpy function that calculates the covariance of the given data
    a = np.cov(data[:,0], data[:,1])

    print("\nCorrelation coefficient (ff)\n")

    print(a[0][0]/(np.std(data[:,0])*np.std(data[:,0])))

    print("\nCorrelation coefficient (fp)\n")

    print(a[0][1]/(np.std(data[:,0])*np.std(data[:,1])))

    print("\nCorrelation coefficient (pf)\n")

    print(a[1][0]/(np.std(data[:,0])*np.std(data[:,1])))

    print("\nCorrelation coefficient (pp)\n")

    print(a[1][1]/(np.std(data[:,1])*np.std(data[:,1])))

    print("\n=====================================================================================\n")

    a = np.cov(data[:,1], data[:,2])

    #print("\nCorrelation coefficient (pp)\n")

    #print(a[0][0]/(np.std(data[:,0])*np.std(data[:,0])))

    print("\nCorrelation coefficient (pl)\n")

    print(a[0][1]/(np.std(data[:,2])*np.std(data[:,1])))

    print("\nCorrelation coefficient (lp)\n")

    print(a[1][0]/(np.std(data[:,2])*np.std(data[:,1])))

    print("\nCorrelation coefficient (ll)\n")

    print(a[1][1]/(np.std(data[:,2])*np.std(data[:,2])))

    print("\n=====================================================================================\n")

    a = np.cov(data[:,2], data[:,0])

    print("\nCorrelation coefficient (lf)\n")

    print(a[0][1]/(np.std(data[:,0])*np.std(data[:,2])))

    print("\nCorrelation coefficient (fl)\n")

    print(a[1][0]/(np.std(data[:,0])*np.std(data[:,2])))

    print("\n=====================================================================================\n")


    