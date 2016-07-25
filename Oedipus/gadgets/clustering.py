#!/usr/bin/python

###################
# Library Imports #
###################
from Oedipus.utils.data import *
from Oedipus.utils.misc import *
from Oedipus.utils.graphics import *
import glob
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy
from numpy.random import randn
from Levenshtein import distance
####################
# Defining Methods #
####################
def agglomerativeClustering(sourceFiles, fileExtension):
    """ Performs agglomerative hierarchical clustering using files with <fileExtension> in the <sourceFiles> directory and return accuracy measure"""
    try:
        accuracy = 0
        # Step 1 - Check the required algorithm to specify the data type to load
        dataFiles = glob.glob("%s/*.%s" % (arguments.sourcedir, arguments.datatype)) # Get the paths of files to load
        dataSamples, dataLabels, loadedClusters = [], [], []
        for dataPoint in dataFiles:
            dataSamples.append([float(x) for x in open(dataPoint).read()[1:-1].split(",")])
            # Also load its cluster
            clusterName, paramNames = loadLabelFromFile(dataPoint.replace(".%s" % arguments.datatype, ".metadata"))
            if not clusterName in loadedClusters:
                loadedClusters.append(clusterName)
            dataLabels.append(loadedClusters.index(clusterName))
        prettyPrint("Successfully retrieved %s instances for clustering" % len(dataSamples))
        # Step 2 - Perform clustering
        clusterer = AgglomerativeClustering(n_clusters=len(loadedClusters))
        predicted = clusterer.fit_predict(numpy.array(dataSamples), dataLabels)
        accuracy = round(metrics.accuracy_score(dataLabels, predicted), 2)

    except Exception as e:
        prettyPrint("Error encountered: %s" % e, "error")

    return accuracy

