'''
Course: EP219 Fall 2017
Course instructor: Vikram Rentala
Assignment number: 1

Description: This program reads the temperature data from 1880 to 2014 (provided by the instructor) and plots some basic histograms following a set of rules

Description of functions:
numpy.ndarray readData(): This method uses the pandas method read_csv to extract data from a .txt file, which is then converted into a numpy array for ease of manipulation which is returned
numpy.ndarray modifyData(data): This method takes the data read by readData and adds the temperature offset to it and puts that data in another column
numpy.ndarray extractYear(data,year = 1912): This method takes a numpy.ndarray containing the data and returns the data pertaining to the specific year given as input (default is 1912)
numpy.ndarray extract15 (data,year = 1912): This method takes a numpy.ndarray containing the data and returns the data pertaining to the 1st and 15th day of every month of a specific year given as input (default is 1912)
void plotHist(data): This method plots a histogram given a set of data

Version: 1.2

Requirements:
Python 2.7 with the following packages installed
    -numpy
    -pandas
    -matplotlib

How to run the code:
In the main function, change hteyear to the year whose data you want to visualize (else it is taken as 1912 by default)
run the code on terminal/any other compiler
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#This method uses the pandas method read_csv to extract data from a .txt file, which is then converted into a numpy array for ease of manipulation which is returned
def readData():
    #reads the data from .txt file and .values converts it to a numpy ndarray
    return pd.read_csv("Complete_TAVG_daily.txt",delim_whitespace=True, header=None).values

#This method takes the data read by readData and adds the temperature offset to it and puts that data in another column
def modifyData(data):
    #creates a new array whose elements are the elements of the 6th column of data + 6.86
    err=data[:,5]+8.68

    #appends the err column to the array
    err = np.reshape(err,(err.size,1))
    x = np.append(data, err, axis=-1)

    return x

#This method takes a numpy.ndarray containing the data and returns the data pertaining to the specific year given as input (default is 1912)
def extractYear(data,year = 1912):

    #boolyears is a numpy array of booleans whose values are decided based on what the following stamenet evaluates to
    booleyears = data[:,1] == year
    #returns numpy array containing only those elements whose booleans correspond to true
    return data[booleyears]

#This method takes a numpy.ndarray containing the data and returns the data pertaining to the 1st and 15th day of every month of a specific year given as input (default is 1912)
def extract15(data,year = 1912):

    #this is a numpy array of booleans whose values are decided based on what the following boolean statement evaluates to
    bool15 = (data[:,1] == year)&((data[:,3] == 1)|(data[:,3] == 15))
    #returns numpy array containing only those elements whose booleans correspond to true
    return data[bool15]

#This method plots a histogram given a set of data
def plotHist(data1, data2):
    #This uses the default hist function to genetate the histogram. The bin size has been set to an optimal value of 12
    plt.subplot(1,2,1)
    plt.hist(data1, 12, alpha = 0.75, rwidth = 0.9)
    plt.xlabel("temperature range")
    plt.ylabel("number of days")
    plt.title("yearly distribution")
    plt.subplot(1,2, 2)
    plt.hist(data2, 12, alpha = 0.75, rwidth = 0.9)
    plt.xlabel("temperature range")
    plt.ylabel("number of days")
    plt.title("distribution at 15 day interval")
    plt.show()

if __name__ == "__main__":

    #Please change this variable to the year whose data you want
    year = 1914
    data = readData()
    print (data)
    data = modifyData(data)
    yearData = extractYear(data, year)
    data15 = extract15(data, year)

    plotHist(yearData[:,6],data15[:,6] )
    means = np.array([0], dtype = float)
    stds = np.array([0], dtype = float)
    #means = np.append(means, [0], axis = -1)
    #print means.shape, "   Maans"
    years = np.array([0], dtype = float)
    for i in range (1880, 2014, 3):
        ydata = extractYear(data, i)
        years = np.append(years, [i], axis = -1)
        mean = np.mean(ydata[:,6])
        std = np.std(ydata[:,6])
        stds = np.append(stds, [std], axis = -1)
        means = np.append(means, [mean], axis = -1)
        #print mean , "   " , i
    print (means)
    plt.plot(years, means, "ob")
    plt.xlim(xmin = 1878, xmax = 2015)
    plt.errorbar(years,means,yerr=stds, linestyle="None")
    plt.show()
