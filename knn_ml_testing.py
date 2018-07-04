# -*- coding: utf-8 -*-
""" 
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
This Machine learning algorithm K-NN wrote by Vytautas Bielinskas 
Data Source: https://archive.ics.uci.edu/ml/machine-learning-databases/00365
Date: 2018-07-04
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
"""
# ::: Importing our Toolkit :::
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ::: Retrieving data :::
def loadFile(path, line):
    df = pd.read_csv(path)
    print(line)
    print('{} file is opened for the model.\n'.format(path))
    
    print('The top of the dataframe is looks like:\n{}'.format(df.head(2)))
    print(type(df))
    print('The dataframe has {} rows and {} columns.\n'.format(df.shape[0], 
          df.shape[1]))
    
    df = df.loc[1 :].replace('?', 'NaN')
    
    return df

# ::: Working with Data (EDA) :::
def minorEDA(df, line):
    
    print(line)
    print('Exploratory Data Analysis is starting...\n')
    print('Data Types, Missing Data, Memory')
    print(df.info())
    
    return None
    
# ::: Set matrix of ML features  :::
def setFeatures(feature, df, line):
    
    y = df[feature]
    X = df.drop(feature, axis = 1)
    return X, y

# ::: Split data-set to Train and Test sets and Plot it :::
def TestTrainFitPlot(X, y, line):
    
    # Cleaning data
    def cleanData(X, line):
        from sklearn.preprocessing import Imputer
    
        print('{}\nCleaning data:'.format(line))
        imp = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
        imp.fit(X)
        
        X = imp.transform(X)
        print('Cleaning is finished.')
        return X  
    
    # Feature scaling
    def featureScaling(X_train, X_test, line):
        print('{}\nFeature scaling start.'.format(line))
        from sklearn.preprocessing import StandardScaler
        
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)
        
        return X_train, X_test
    
    # Plot the results
    def plotAccuracies(neighbors, train_accuracy, test_accuracy, line):
        print('{}\nPloting results of K-NN for current data-set.'.format(line))
        
        plt.title('K-NN Neighbors')
        plt.xlabel('Neighbors\n(#)')
        plt.ylabel('Accuracy\n(%)', rotation = 0, labelpad = 35)
        plt.plot(neighbors, test_accuracy, label = 'Test Accuracy')
        plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
        
        for a, b in zip(neighbors, test_accuracy):
            plt.text(a, b, str(round(b, 2)))
            
        plt.legend()
        
        plt.grid(which = 'major')
        
        plt.show()
        
        return None
    
    print('{}\nMachine learning part:'.format(line))
    
    neighbors = np.arange(1, 10)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
    
    # Data pre-proceesing: Clean the data
    X = cleanData(X, line)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,
                                                        random_state = 42,
                                                        stratify = y)
    
    # >> Feature Scaling
    X_train, X_test = featureScaling(X_train, X_test, line)
    
    # >> Set the K-NN classifier
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    
    # >> Get Default Training Accuracy
    print('Default Accuracy: {}.'.format(round(knn.score(X_test, y_test), 3)))
    
    # >> Check Accuracy with other values of neighbors
    for acc, n in enumerate(neighbors):
        knn = KNeighborsClassifier(n_neighbors = n)
        knn.fit(X_train, y_train)
        train_accuracy[acc] = knn.score(X_train, y_train)
        test_accuracy[acc] = knn.score(X_test, y_test)
    
    print('{}\nWe got following test:\n{}'.format(line, test_accuracy))
    
    # >> Plot the results of K-NN
    plotAccuracies(neighbors, train_accuracy, test_accuracy, line)
    
    return None

"""
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:::                THE FRAME OF MACHINE LEARNING HERE                       ::: 
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
"""
# ::: Set the current working directory :::
import os, glob

cwd = os.getcwd()
os.chdir(cwd)
extension = 'csv'

print('The current working directory is {}.'.format(cwd))
result = [i for i in glob.glob('*.{}'.format(extension))]
print('In total {} *.{} files found on the working directory:\n{}.\n'
      .format(len(result), extension, result))

# Set the number of a file to be read in working directory
number_of_file = 4
line = '-' * 60

if __name__ == '__main__':
    df = loadFile('\\'.join([cwd, result[0]]), line)
    minorEDA(df, line)
    X, y = setFeatures('class', df, line)
TestTrainFitPlot(X, y, line)
