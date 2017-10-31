'''
Course: EP219 Fall 2017
Course instructor: Vikram Rentala
Assignment number: 4

Description: This program reads the temperature data from 1880 to 2014 (provided by the instructor) and plots the mean temperature of a each year (plotted in samples of 3) along with the errors in the average temperature 
             It also tries to fit the data to a line and to a quadraic curve
             It then plots the contour map of the error function (if the chi-squared distribution)
             It also verifies the minima in the error functions with respect to each of the parameters

Description of functions:
numpy.ndarray readData(): This method uses the pandas method read_csv to extract data from a .txt file, which is then converted into a numpy array for ease of manipulation which is returned
numpy.ndarray modifyData(data): This method takes the data read by readData and adds the temperature offset to it and puts that data in another column
numpy.ndarray extractYear(data,year = 1912): This method takes a numpy.ndarray containing the data and returns the data pertaining to the specific year given as input (default is 1912)


Version: 1.0

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
import math
import scipy.optimize as optimize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

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


if __name__ == "__main__":

    #Please change this variable to the year whose data you want
    year = 1914
    data = readData()
    data = modifyData(data)
    yearData = extractYear(data, year)

    means = np.array([0], dtype = float)
    stds = np.array([0], dtype = float)
    stds1 = np.array([0], dtype = float)
    years = np.array([0], dtype = float)

    for i in range (1880, 2014, 3):
        ydata = extractYear(data, i)
        years = np.append(years, [i], axis = -1)
        mean = np.mean(ydata[:,6])
        std = np.std(ydata[:,6])
        stds = np.append(stds, [std], axis = -1)

        #Estimating the error in the average temperature
        #Taking into account the leap year
        if(i%4 == 0):
            std1 = math.sqrt(std*std/365)
            stds1 = np.append(stds1, [std1], axis = -1)

        #The other non-leap years
        else:
            std1 = math.sqrt(std*std/364)
            stds1 = np.append(stds1, [std1], axis = -1)

        means = np.append(means, [mean], axis = -1)

    ax = plt.subplot(1,1,1)
    plt.plot(years, means, "ob")
    plt.xlim(xmin = 1878, xmax = 2015)
    print(stds.shape)
    print(stds1.shape)

    #The earlier error bars
    plt.errorbar(years,means,yerr=stds, ecolor = 'g')
    #The new error bars
    plt.errorbar(years,means,yerr=stds1, ecolor='r', capthick=2)

    #Fitting the data to a straight line
    coeffs = np.polyfit(years, means,1)

    #The optimum values of M and C to minimize the error
    print(coeffs)
    polynomial = np.poly1d(coeffs)
    ys = polynomial(years)
    plt.plot(years, ys, label = "Linear fit")

    #Fitting the data to a quadratic
    coeffs = np.polyfit(years, means,2)
    #The optimum values of the coefficients of the quadratic to minimize the error
    print(coeffs)
    polynomial = np.poly1d(coeffs)

    ys = polynomial(years)
    plt.plot(years, ys, label = 'Quadratic fit')
    #m and c min came out to be 0.0045, 0.1
    #plt.plot(np.array([1880,2010]),np.array([8.2,9.145]), label = "Our estimate")
    #c = -7.3, m = 9.0413
    plt.plot(np.array([1880,2010]),np.array([7.9844,9.043]), label = "Our estimate")
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels)
    plt.xlabel("Years")
    plt.ylabel("Temperature")
    
    plt.show()

    Ti = means
    Yi = np.array([i for i in range (1880, 2014, 3)])
    means = means[1:]

    
    fig = plt.figure()

    

    #Defining the parameter space for plotting the contours
    #The limits for the parameter space were decided by taking hints from the optimal values predicted by the linear fitting function
    #These specific values were set after a lot of tweaking to get a nice informative contour

    """
    C1 = np.arange(-0.2, 0.1, 0.02)
    M1 = np.arange(0.00432, 0.00462, 0.0001)
    

    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Change the Ranges in C1 and M1
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    C1 = np.arange(-10.16, -10.12, 0.0001)
    print(C1.shape)
    M1 = np.arange(0.00965, 0.00969, 0.000005)
    print(M1.shape)
    
    C, M = np.meshgrid(C1, M1)
    Ti = Ti[1:]
    stds = stds[1:]

    #Defining the chi-square error function
    T1 = np.sum((Ti/stds)**2)
    T2 = M*M*np.sum((Yi/stds)**2)
    T3 = C*C*np.sum((1/stds)**2)
    T4 = -2*M*np.sum(Yi*Ti/(stds)**2)
    T5 = 2*M*C*np.sum(Yi/(stds)**2)
    T6 = -2*C*np.sum(Ti/(stds)**2)

    Z = T1 + T2 + T3 + T4 + T5 + T6
    Z = Z/44 #per degree of freedom 46 - 2

    print(np.amin(Z),"minimum value")

    minval = Z[0][0]
    minind = (0,0)
    for (x,y), value in np.ndenumerate(Z):
        if Z[x][y] < minval:
            minind = (x,y)
            minval = Z[x][y]
    print(minval, "minimum value")
    print(minind, "Index of minimum values")
    print(Z[minind[0]][minind[1]] , "Z of minimum values")
    print(M1[minind[0]], C1[minind[1]] )


    """

    K = np.ndarray.tolist(Z)

    initial_guess = [1, 1]
    result = optimize.minimize(K, initial_guess)
    if result.success:
        fitted_params = result.x
        print(fitted_params)
    else:
        raise ValueError(result.message)

    """
    #print(Z.shape)
    #print(C.shape)
    #print(M.shape)
    print(type(Z))
    #print(Z)

    #Plotting the contour map
    CS = plt.contour(C, M, Z, 200)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Contour plot for minimum error by varying M and C')
    plt.ylabel("m (slope)")
    plt.xlabel("c (x-intercept)")
    plt.show()


    

    """
    The 1 - D Xhi - squared plots

    """


    M = np.arange(0.00965, 0.00969, 0.000005)
    #c = -0.18095889
    c = -10.1411
    T1 = np.sum((Ti/stds)**2)
    T2 = M*M*np.sum((Yi/stds)**2)
    T3 = c*c*np.sum((1/stds)**2)
    T4 = -2*M*np.sum(Yi*Ti/(stds)**2)
    T5 = 2*M*c*np.sum(Yi/(stds)**2)
    T6 = -2*c*np.sum(Ti/(stds)**2)
    Z = T1 + T2 + T3 + T4 + T5 + T6
    plt.plot(M,Z/44)
    plt.title("Plot of Error vs M - Putting C at its minimum value")
    plt.xlabel("M")
    plt.ylabel("Error")
    plt.show()

    C = np.arange(-10.16, -10.12, 0.0001)

    #m = 0.00454967
    m = 0.009665
    T1 = np.sum((Ti/stds)**2)
    T2 = m*m*np.sum((Yi/stds)**2)
    T3 = C*C*np.sum((1/stds)**2)
    T4 = -2*m*np.sum(Yi*Ti/(stds)**2)
    T5 = 2*m*C*np.sum(Yi/(stds)**2)
    T6 = -2*C*np.sum(Ti/(stds)**2)
    Z = T1 + T2 + T3 + T4 + T5 + T6
    plt.plot(C,Z/44)
    plt.title("Plot of Error vs C - Putting M at its minimum value")
    plt.xlabel("C")
    plt.ylabel("Error")
    plt.show()