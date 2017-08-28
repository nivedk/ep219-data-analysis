import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#This method uses the pandas method read_csv to extract data from a .txt file, which is then converted into a numpy array for ease of manipulation which is returned
def readData():
    #reads the data from .txt file and .values converts it to a numpy ndarray
    return pd.read_csv("facebookdata.csv", header=None).values

if __name__ == "__main__":
    data = readData()
    #print(a)
    #print (type(a))
    #print (a.shape)
    data = (data[1:,:])
    data = data.astype(np.float)
    print (data)
    plt.xlabel("number of friends ")
    plt.ylabel("number of people")
    plt.title("distribution of facebook friends")
    plt.hist(data[:,0], alpha = 0.75, rwidth = 0.9)
    plt.axvline(np.mean(data[:,0]), color='g', linestyle='dashed', linewidth=2)
    plt.axvline(np.mean(data[:,0]) + np.std(data[:,0]), color='r', linestyle='dashed', linewidth=2)
    plt.axvline(np.mean(data[:,0]) - np.std(data[:,0]), color='r', linestyle='dashed', linewidth=2)
    print(type(data[0,0]))
    #plt.hist([1, 2, 3,4,5,3,2,1,6,7,8,5,4,3,2,5],12, alpha = 0.75, rwidth = 0.9)
    plt.show()
    plt.xlabel("number of posts ")
    plt.ylabel("number of people")
    plt.title("distribution of facebook posts")
    plt.hist(data[:,1], alpha = 0.75, rwidth = 0.9)
    plt.axvline(np.mean(data[:,1]), color='g', linestyle='dashed', linewidth=2)
    plt.axvline(np.mean(data[:,1]) + np.std(data[:,1]), color='r', linestyle='dashed', linewidth=2)
    plt.axvline(np.mean(data[:,1]) - np.std(data[:,1]), color='r', linestyle='dashed', linewidth=2)
    plt.show()
    plt.xlabel("number of likes ")
    plt.ylabel("number of people")
    plt.title("distribution of facebook likes")
    plt.hist(data[:,2], alpha = 0.75, rwidth = 0.9)
    plt.axvline(np.mean(data[:,2]), color='g', linestyle='dashed', linewidth=2)
    plt.axvline(np.mean(data[:,2]) + np.std(data[:,2]), color='r', linestyle='dashed', linewidth=2)
    plt.axvline(np.mean(data[:,2]) - np.std(data[:,2]), color='r', linestyle='dashed', linewidth=2)
    plt.show()
    plt.hexbin(data[:,0], data[:,1])
    plt.show()
    plt.hexbin(data[:,1], data[:,2])
    plt.show()
    plt.hexbin(data[:,2], data[:,0])
    plt.show()
