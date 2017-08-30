'''
Course: EP219 Fall 2017
Course instructor: Vikram Rentala
Assignment number: 2

Description: This program reads the 10,000 anonymous facebook user data (containing number of friends, number of posts, number of likes) collected over a period of 10 years (provided by the instructor) and plots 1D and 2D histograms to aid in visualizing the data

Description of functions:
numpy.ndarray readData(): This method uses the pandas method read_csv to extract data from a .csv file, which is then converted into a numpy array for ease of manipulation which is returned
main : In this code, most of the processing happens in the main function

Version: 1.0

Requirements:
Python 2.7 with the following packages installed
    -numpy
    -pandas
    -matplotlib

LaTeX which is configured to run with python (in case this is not present, some of the labels etc may not be visible)

How to run the code:
Make sure, the facebookdata.csv file is in the same direct folder as this file (assn2.py), and just run it on terminal/command prompt or any other IDE/program
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
    return pd.read_csv("facebookdata.csv", header=None).values

if __name__ == "__main__":
    data = readData()
    data = (data[1:,:])
    data = data.astype(np.float)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#CODE FOR PLOTTING 1D HISTOGRAMS 

    #plotting the 1D histogram showing the distribution of facebook friends
    ax = plt.subplot(1,1,1)
    #Setting the label of the x-axis
    plt.xlabel("number of friends ")
    #Setting the label of the y-axis
    plt.ylabel("number of people")
    #alpha sets the transparency and rwidth sets the distance between two bins
    plt.title('distribution of facebook friends')
    plt.hist(data[:,0], alpha = 0.75, rwidth = 0.9)
    #Calculating and plotting vertical lines for mean and standard deviation on either side
    plt.axvline(np.mean(data[:,0]), color='g', linestyle='dashed', linewidth=2, label = "Mean " + (str)(np.mean(data[:,0])))
    plt.axvline(np.mean(data[:,0]) + np.std(data[:,0]), color='r', linestyle='dashed', linewidth=2, label = "Standard Deviation \n Mean +/- " + (str)("%.4f"%np.std(data[:,0])))
    plt.axvline(np.mean(data[:,0]) - np.std(data[:,0]), color='r', linestyle='dashed', linewidth=2)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels)
    plt.show()

    #plotting the 1D histogram showing the distribution of facebook posts
    ax = plt.subplot(1,1,1)
    #Setting the label of the x-axis
    plt.xlabel("number of posts ")
    #Setting the label of the y-axis
    plt.ylabel("number of people")
    plt.title("distribution of facebook posts")
    #alpha sets the transparency and rwidth sets the distance between two bins
    plt.hist(data[:,1], alpha = 0.75, rwidth = 0.9)
    #Calculating and plotting vertical lines for mean and standard deviation on either side
    plt.axvline(np.mean(data[:,1]), color='g', linestyle='dashed', linewidth=2, label = "Mean " + (str)(np.mean(data[:,1])))
    plt.axvline(np.mean(data[:,1]) + np.std(data[:,1]), color='r', linestyle='dashed', linewidth=2, label = "Standard Deviation \n Mean +/- " + (str)("%.4f"%np.std(data[:,1])))
    plt.axvline(np.mean(data[:,1]) - np.std(data[:,1]), color='r', linestyle='dashed', linewidth=2)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels)
    plt.show()

    #plotting the 1D histogram showing the distribution of facebook likes
    ax = plt.subplot(1,1,1)
    #Setting the label of the x-axis
    plt.xlabel("number of likes ")
    #Setting the label of the y-axis
    plt.ylabel("number of people")
    plt.title("distribution of facebook likes")
    #alpha sets the transparency and rwidth sets the distance between two bins
    plt.hist(data[:,2], alpha = 0.75, rwidth = 0.9)
    #Calculating and plotting vertical lines for mean and standard deviation on either side
    plt.axvline(np.mean(data[:,2]), color='g', linestyle='dashed', linewidth=2, label = "Mean " + (str)(np.mean(data[:,2])))
    plt.axvline(np.mean(data[:,2]) + np.std(data[:,2]), color='r', linestyle='dashed', linewidth=2, label = "Standard Deviation \n Mean +/- " + (str)("%.4f"%np.std(data[:,2])))
    plt.axvline(np.mean(data[:,2]) - np.std(data[:,2]), color='r', linestyle='dashed', linewidth=2)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels)
    plt.show()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#CODE FOR PLOTTING 2D HISTOGRAMS

#The bin size has been chosen to an optimal value of 100.
#For values larger than 100, the plot becomes noisy as even in dense areas, there are small blocks where there are no values, thus making it hard to visualize the data
#For values smaller than 100, the continuity in the data is not clear. 


    #Including Latex formatting 
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    #Plotting the 2D histogram, number of bins is 100 and the color map used is plasma
    plt.hist2d(data[:,0], data[:,1], bins=100, cmap = 'plasma')
    #labelling the axes
    plt.xlabel(r'\textbf{\Large friends  \textrightarrow}')
    plt.ylabel(r'\textbf{\Large posts  \textrightarrow}')
    plt.title(r'{\huge \textbf{posts} {\Large vs} \textbf{friends}}')
    plt.colorbar()
    plt.show()

    #Including Latex formatting 
    plt.rc('text', usetex=True)
    #Plotting the 2D histogram, number of bins is 100 and the color map used is plasma
    plt.hist2d(data[:,1], data[:,2], bins=100, cmap = 'plasma')
    #labelling the axes
    plt.xlabel(r'\textbf{\Large posts  \textrightarrow}')
    plt.ylabel(r'\textbf{\Large likes  \textrightarrow}')
    plt.title(r'{\huge \textbf{likes} {\Large vs} \textbf{posts}}')
    plt.colorbar()
    plt.show()

    #Including Latex formatting 
    plt.rc('text', usetex=True)
    #Plotting the 2D histogram, number of bins is 100 and the color map used is plasma
    plt.hist2d(data[:,2], data[:,0], bins=100, cmap = 'plasma')
    #labelling the axes
    plt.xlabel(r'\textbf{\Large likes  \textrightarrow}')
    plt.ylabel(r'\textbf{\Large friends  \textrightarrow}')
    plt.title(r'{\huge \textbf{friends} {\Large vs} \textbf{likes}}')
    plt.colorbar()
    plt.show()
