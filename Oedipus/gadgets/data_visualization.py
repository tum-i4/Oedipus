#!/usr/bin/python

###################
# Library imports #
###################
from Oedipus.utils.data import *
from Oedipus.utils.misc import *
from Oedipus.utils.graphics import *
import matplotlib.pyplot as pyplot
import pylab as P
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import glob
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy
import time
####################
# Defining Methods #
####################
def plotAccuracyGraph(X, Y, Xlabel='Variable', Ylabel='Accuracy', graphTitle="Test Accuracy Graph", filename="graph.pdf"):
    """ Plots and saves accuracy graphs """
    try:
        timestamp = int(time.time())
        fig = P.figure(figsize=(8,5))
        # Set the graph's title
        P.title(graphTitle, fontname='monospace')
        # Set the axes labels
        P.xlabel(Xlabel, fontsize=12, fontname='monospace')
        P.ylabel(Ylabel, fontsize=12, fontname='monospace')
        # Add horizontal and vertical lines to the graph
        P.grid(color='DarkGray', linestyle='--', linewidth=0.1, axis='both')
        # Add the data to the graph
        P.plot(X, Y, 'r-*', linewidth=1.0)
        # Save figure
        prettyPrint("Saving figure to ./%s" % filename)#(graphTitle.replace(" ","_"), timestamp))
        P.tight_layout()
        fig.savefig("./%s" % filename)#(graphTitle.replace(" ", "_"), timestamp))
        
    except Exception as e:
        prettyPrint("Error encountered in \"plotAccuracyGraph\": %s" % e, "error")
        return False

    return True
 
def plotReductionGraph(dataSamples, dataLabels, classNames, dimension=2, graphTitle="Test Graph", filename="reduction.pdf"):
    """ Plots data sample visualization graphs """
    try:
        timestamp = int(time.time())
        colors = ['DarkRed', 'DarkGreen', 'DarkBlue', 'DarkOrange', 'DarkMagenta', 'DarkCyan', 'Gray', 'Black']
        randomColor = lambda: random.randint(0,255)
        markers = ['*', 'o', 'v', '^', 's', 'd', 'D', 'p', 'h', 'H', '<', '>', '.', ',', '|', '_']

        fig = P.figure(figsize=(8,5))
        if dimension == 3:
            ax = fig.add_subplot(111, projection='3d')
        P.title(graphTitle, fontname='monospace')
        if dimension == 2:
            P.xlabel('x1', fontsize=12, fontname='monospace')
            P.ylabel('x2', fontsize=12, fontname='monospace')
        else:
            ax.set_xlabel('x1', fontsize=12, fontname='monospace')
            ax.set_ylabel('x2', fontsize=12, fontname='monospace')
            ax.set_zlabel('x3', fontsize=12, fontname='monospace')

        P.grid(color='DarkGray', linestyle='--', linewidth=0.1, axis='both')
       
        for c in range(len(classNames)):
            X,Y,Z = [], [], []
            for labelIndex in range(len(dataLabels)):
                if c == dataLabels[labelIndex]:
                    X.append(dataSamples[labelIndex,:].tolist()[0])
                    Y.append(dataSamples[labelIndex,:].tolist()[1])
                    if dimension == 3:
                        Z.append(dataSamples[labelIndex,:].tolist()[2])

            # Plot points of that class
            #P.plot(Y, X, color='#%02X%02X%02X' % (randomColor(), randomColor(), randomColor()), marker=markers[c], markeredgecolor='None', markersize=4.0, linestyle='None', label=classNames[c])
            if dimension == 2:
                P.plot(Y, X, color=colors[c % len(colors)], marker=markers[c % len(markers)], markersize=5.0, linestyle='None', label=classNames[c])
            else:
                ax.scatter(X,Y,Z,c=colors[c % len(colors)], marker=markers[c % len(markers)])
                
        if dimension == 2:
            #P.legend([x.split(",")[-1] for x in classNames], fontsize='xx-small', numpoints=1, fancybox=True)
            P.legend([x for x in classNames], fontsize='xx-small', numpoints=1, fancybox=True)
        else:
            ax.legend([x for x in classNames], fontsize='xx-small', numpoints=1, fancybox=True)

        prettyPrint("Saving results to ./%s" % filename)#(graphTitle, timestamp))
        P.tight_layout()
        fig.savefig("./%s" % filename)#(graphTitle, timestamp))

    except Exception as e:
        prettyPrint("Error encountered in \"plotReductionGraph\": %s" % e, "error")
        return False
    
    return True
    
def visualizeData(sourceDir, fileExtension, dimension, algorithm="tsne", filename="visualization.pdf"):
    """ Reduces data to <dimension>-dimensional space using PCA and plots results """
    try:
        prettyPrint("Loading data samples of extension \"%s\" from \"%s\"" % (fileExtension, sourceDir))
        dataFiles = sorted(glob.glob("%s/*.%s" % (sourceDir, fileExtension)))
        dataSamples, dataLabels, loadedClasses = [], [], []
        for dataPoint in dataFiles:
            if fileExtension == "triton":
                # For "triton" features that contain numerical/nominal features
                allAttributes = open(dataPoint).read().replace("\n", "").replace(" ", "")[1:-1].split(",")
                features = []
                for attribute in allAttributes:
                    if attribute.lower().find("yes") != -1:
                        features.append(1)
                    elif attribute.lower().find("no") != -1:
                        features.append(0)
                    else:
                        features.append(attribute)
                dataSamples.append(features)
            else:
                # For all other extensions that contain numerical features
                dataSamples.append([float(x) for x in open(dataPoint).read()[1:-1].replace(" ","").split(",")])
            # Also load its class
            className, paramNames = loadLabelFromFile(dataPoint.replace(".%s" % fileExtension, ".label"))
            clusterName = className.split(",")[-1]# + str(paramNames)
            if not clusterName in loadedClasses:
                loadedClasses.append(clusterName)
            dataLabels.append(loadedClasses.index(clusterName))
                
        prettyPrint("Successfully loaded %s samples from \"%s\"" % (len(dataSamples),sourceDir))
            
        # Perform PCA or feature selection
        prettyPrint("Projecting feature vectors into %s-dimensional space using %s" % (dimension, algorithm))
        dataSamples = numpy.array(dataSamples) # Convert list to a numpy array
        if algorithm == "tsne":
            tsne = TSNE(n_components=int(dimension), random_state=0)
            newdataSamples = tsne.fit_transform(dataSamples)
        else:
            pca = PCA(n_components = int(dimension))
            newdataSamples = pca.fit_transform(dataSamples)

        # Step 4 - Plot data
        prettyPrint("Plotting data using \"%s\"" % algorithm)
        plotReductionGraph(newdataSamples, dataLabels, loadedClasses, dimension=int(dimension), graphTitle="%s-d_%s-features with %s" % (dimension, fileExtension, algorithm), filename="%s_d_%s_%s.pdf" % (dimension, fileExtension, algorithm))
        
    except Exception as e:
        prettyPrint("Error encountered: %s" % e, "error")
        return False
     
    return True     

def visualizeOriginal(sourceDir, fileExtension, dimension, algorithm="tsne", filename="original.pdf"):
    """ Plots original programs against their obfuscated versions in a <dimension>-dimensional space using PCA """
    try:
        prettyPrint("Loading the list of original programs")
        originalFiles = list(set(sorted(glob.glob("%s/*.c" % sourceDir))) - set(sorted(glob.glob("%s/*_*.c" % sourceDir))))
        prettyPrint("Successfully retrieved %s original programs" % len(originalFiles))
        # Retrieve obfuscated versions of every program and plot them
        for program in originalFiles:
            obfuscatedPrograms = glob.glob("%s*_*.c" % program.replace(".c", ""))
            prettyPrint("Successfully retrieved %s obfuscated versions of \"%s\"" % (len(obfuscatedPrograms), program), "debug")
            # Retrieve the data points to plot
            dataFiles = [] + obfuscatedPrograms + [program]
            prettyPrint("Loading data samples of extension \"%s\" from \"%s\"" % (fileExtension, sourceDir))
            dataSamples, dataLabels, loadedClasses = [], [], []
            dataFiles = [x.replace(".c", ".%s" % fileExtension) for x in dataFiles]
        for dataPoint in dataFiles:
            dataSamples.append([float(x) for x in open(dataPoint).read()[1:-1].split(",")])
            # Also load its class
            className, paramNames = loadLabelFromFile(dataPoint.replace(".%s" % fileExtension, ".label"))
            clusterName = className.split(",")[-1]# + str(paramNames)
            if not clusterName in loadedClasses:
                loadedClasses.append(clusterName)
            dataLabels.append(loadedClasses.index(clusterName))

        prettyPrint("Successfully loaded %s samples from \"%s\"" % (len(dataSamples), sourceDir)) 

        # Use PCA to reduce the dimensionality of data points
        prettyPrint("Projecting feature vectors into %s-dimensional space" % dimension)
        dataSamples = numpy.array(dataSamples) # Convert list to a numpy array
        if algorithm == "tsne":
            tsne = TSNE(n_components = int(dimension), random_state=0)
            newdataSamples = tsne.fit_transform(dataSamples)
        else:
            pca = PCA(n_components = int(dimension))
            newdataSamples = pca.fit_transform(dataSamples)

        # Step 4 - Plot data
        prettyPrint("Plotting data using \"PCA\"")
        plotReductionGraph(newdataSamples, dataLabels, loadedClasses, dimension=int(dimension), graphTitle="%s-d %s-features for %s" % (dimension, fileExtension, program.replace(sourceDir, "")), filename="%s_d_%s_%s.pdf" % (dimension, fileExtension, algorithm))
        time.sleep(1) # Allow for figures to be rendered
    except Exception as e:
        prettyPrint("Error encountered: %s" % e, "error")
        return False
    
    return True
