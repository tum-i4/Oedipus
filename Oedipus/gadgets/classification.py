#!/usr/bin/python

###################
# Library Imports #
###################
from Oedipus.utils.data import *
from Oedipus.utils.misc import *
from Oedipus.utils.graphics import *
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB # Multinomial Naive Bayes
import numpy
from numpy.random import randn
# For tree visualization
from sklearn.externals.six import StringIO
import pydot
import sys, os
#####################
# Utility Functions #
#####################
def numOfMismatches(s1, s2):
    """ Returns number of character mismatches in two strings """
    s1Letters = {k: s1.count(k) for k in s1}
    s2Letters = {k: s2.count(k) for k in s2}
    # Compare matches
    s = {}
    for k2 in s2Letters:
        if k2 in s1Letters.keys():
           s[k2] = abs(s1Letters[k2] - s2Letters[k2])
        else:
           s[k2] = s2Letters[k2]
    # Sum up remaining matches
    mismatches = sum(s.values())
    return mismatches

def findTrend(trend, trendList):
    """ Finds a specific trend tuple within a list of tuples """
    # Assuming the lack of duplicates
    for tIndex in range(len(trendList)):
        if len(trendList[tIndex]) == 2:
            if trendList[tIndex][0] == trend:
                return tIndex, trendList[tIndex]
    return -1, ()

def mergeTrends(oldTrends, newTrends):
    """ Merges two lists of trend tuples, updating the count on-the-fly """
    tempTrends = [] + oldTrends
    for tIndex in range(len(newTrends)):
        trend = newTrends[tIndex]
        oldTrendIndex, oldTrend = findTrend(trend[0], tempTrends)
        if oldTrendIndex != -1 and len(oldTrend) > 0:
            nTrend = (oldTrend[0], oldTrend[1]+trend[1])
            tempTrends.pop(oldTrendIndex)
            tempTrends.append(nTrend)
        else:
            tempTrends.append(trend)

    return tempTrends

def cmpTuple(x,y):
    """ Compares two tuples to the end of sorting a list of tuples """
    if x[1] > y[1]:
        return -1
    elif x[1] < y[1]:
        return 1
    else:
        return 0

##################
# Main functions #
##################
def classifyNaiveBayes(Xtr, ytr, Xte, yte, reduceDim="none", targetDim=0):
    """ Classified data using Naive Bayes """
    try:
        accuracyRate, timing, probabilities = 0.0, 0.0, []
        # Reduce dimensionality if requested
        Xtr = reduceDimensionality(Xtr, ytr, reduceDim, targetDim) if reduceDim != "none" else Xtr
        Xte = reduceDimensionality(Xte, yte, reduceDim, targetDim) if reduceDim != "none" else Xte
        # Make sure values are positive because MultinomialNB doesn't take negative features
        Xtr = flipSign(Xtr, "+")
        Xte = flipSign(Xte, "+")
        # Perform classification
        nbClassifier = MultinomialNB()
        prettyPrint("Training the Naive Bayes algorithm", "debug")
        startTime = time.time()
        nbClassifier.fit(numpy.array(Xtr), numpy.array(ytr))
        # Now test the trained algorithm
        prettyPrint("Submitting the test samples", "debug")
        predicted = nbClassifier.predict(Xte)
        endTime = time.time()
        # Compare the predicted and ground truth
        accuracyRate = round(metrics.accuracy_score(predicted, yte), 2)
        probabilities = nbClassifier.predict_proba(Xte)
        # Finally, calculate the time taken to train and classify
        timing = endTime-startTime

    except Exception as e:
        prettyPrint("Error encountered in \"classifyNaiveBayes\": %s" % e, "error")
    
    return accuracyRate, timing, probabilities, predicted


def classifyNaiveBayesKFold(X, y, kFold=2, reduceDim="none", targetDim=0):
    """ Classifies data using Naive Bayes and K-Fold cross validation """
    try:
        groundTruthLabels, predictedLabels = [], []
        accuracyRates = [] # Meant to hold the accuracy rates
        # Split data into training and test datasets
        trainingDataset, testDataset = [], []
        trainingLabels, testLabels = [], []
        accuracyRates = []
        probabilities = []
        timings = []
        # Reduce dimensionality if requested
        if reduceDim != "none":
            X_new = reduceDimensionality(X, y, reduceDim, targetDim)
        else:
            X_new = X
        # Now carry on with classification
        kFoldValidator = KFold(n=len(X_new), n_folds=kFold, shuffle=False)
        # Make sure values are positive because MultinomialNB doesn't take negative features
        X_new = flipSign(X_new, "+")
        for trainingIndices, testIndices in kFoldValidator:
            # Prepare the training and testing datasets
            for trIndex in trainingIndices:
                trainingDataset.append(X_new[trIndex])
                trainingLabels.append(y[trIndex])
            for teIndex in testIndices:
                testDataset.append(X_new[teIndex])
                testLabels.append(y[teIndex])
            # Perform classification
            startTime = time.time()
            nbClassifier = MultinomialNB()
            prettyPrint("Training the Naive Bayes algorithm", "debug")
            nbClassifier.fit(numpy.array(trainingDataset), numpy.array(trainingLabels))
            prettyPrint("Submitting test samples", "debug")
            predicted = nbClassifier.predict(testDataset)
            endTime = time.time()
            # Add that to the groundTruthLabels and predictedLabels matrices
            groundTruthLabels.append(testLabels)
            predictedLabels.append(predicted)
            # Compare the predicted and ground truth and appent to list
            accuracyRates.append(round(metrics.accuracy_score(predicted, testLabels), 2))
            # Also append the probability estimates
            probs = nbClassifier.predict_proba(testDataset)
            probabilities.append(probs)
            timings.append(endTime-startTime) # Keep track of performance

            trainingDataset, trainingLabels = [], []
            testDataset, testLabels = [], []

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        prettyPrint("Error encountered in \"classifyNaiveBayesKFold\" +%s: %s" % (exc_tb.tb_lineno, e), "error")
        return [], [], []

    return accuracyRates, probabilities, timings, groundTruthLabels, predictedLabels

def classifyTree(Xtr, ytr, Xte, yte, splitCriterion="gini", maxDepth=0, visualizeTree=False):
    """ Classifies data using CART """
    try:
        accuracyRate, probabilities, timing = 0.0, [], 0.0
        # Perform classification
        cartClassifier = tree.DecisionTreeClassifier(criterion=splitCriterion, max_depth=maxDepth)
        startTime = time.time()
        prettyPrint("Training a CART tree for classification using \"%s\" and maximum depth of %s" % (splitCriterion, maxDepth), "debug")
        cartClassifier.fit(numpy.array(Xtr), numpy.array(ytr))
        prettyPrint("Submitting the test samples", "debug")
        predicted = cartClassifier.predict(Xte)
        endTime = time.time()
        # Compare the predicted and ground truth and append result to list
        accuracyRate = round(metrics.accuracy_score(predicted, yte), 2)
        # Also append the probability estimates
        probs = cartClassifier.predict_proba(Xte)
        probabilities.append(probs)
        timing = endTime-startTime # Keep track of performance
        if visualizeTree:
            # Visualize the tree
            dot_data = StringIO()
            tree.export_graphviz(cartClassifier, out_file=dot_data)
            graph = pydot.graph_from_dot_data(dot_data.getvalue())
            prettyPrint("Saving learned CART to \"tritonTree_%s.pdf\"" % getTimestamp(), "debug")
            graph.write_pdf("tree_%s.pdf" % getTimestamp())
  
    except Exception as e:
        prettyPrint("Error encountered in \"classifyTree\": %s" % e, "error")

    return accuracyRate, timing, probabilities, predicted

def classifyTreeKFold(X, y, kFold=2, splitCriterion="gini", maxDepth=0, visualizeTree=False):
    """ Classifies data using CART and K-Fold cross validation """
    try:
        groundTruthLabels, predictedLabels = [], []
        accuracyRates = [] # Meant to hold the accuracy rates
        # Split data into training and test datasets
        trainingDataset, testDataset = [], []
        trainingLabels, testLabels = [], []
        accuracyRates = []
        probabilities = []
        timings = []
        kFoldValidator = KFold(n=len(X), n_folds=kFold, shuffle=False)
        currentFold = 1
        for trainingIndices, testIndices in kFoldValidator:
            # Prepare the training and testing datasets
            for trIndex in trainingIndices:
                trainingDataset.append(X[trIndex])
                trainingLabels.append(y[trIndex])
            for teIndex in testIndices:
                testDataset.append(X[teIndex])
                testLabels.append(y[teIndex])
            # Perform classification
            startTime = time.time()
            cartClassifier = tree.DecisionTreeClassifier(criterion=splitCriterion, max_depth=maxDepth)
            prettyPrint("Training a CART tree for classification using \"%s\" and maximum depth of %s" % (splitCriterion, maxDepth), "debug")
            cartClassifier.fit(numpy.array(trainingDataset), numpy.array(trainingLabels))
            prettyPrint("Submitting the test samples", "debug")
            predicted = cartClassifier.predict(testDataset)
            endTime = time.time()
            # Add that to the groundTruthLabels and predictedLabels matrices
            groundTruthLabels.append(testLabels)
            predictedLabels.append(predicted)
            # Compare the predicted and ground truth and append result to list
            accuracyRates.append(round(metrics.accuracy_score(predicted, testLabels), 2))
            # Also append the probability estimates
            probs = cartClassifier.predict_proba(testDataset)
            probabilities.append(probs)
            timings.append(endTime-startTime) # Keep track of performance
            if visualizeTree:
                # Visualize the tree
                dot_data = StringIO() 
                tree.export_graphviz(cartClassifier, out_file=dot_data) 
                graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
                prettyPrint("Saving learned CART to \"tritonTree_%s.pdf\"" % currentFold, "debug")
                graph.write_pdf("tritonTree_%s.pdf" % currentFold)

            trainingDataset, trainingLabels = [], []
            testDataset, testLabels = [], []
            currentFold += 1 

    except Exception as e:
        prettyPrint("Error encountered in \"classifyTreeKFold\": %s" % e, "error")
        return [], [], []

    return accuracyRates, probabilities, timings, groundTruthLabels, predictedLabels

def reduceDimensionality(X, y, method="selectkbest", targetDim=10):
    """ Reduces the dimensionality of [X] to [targetDim] """
    try:
        # Check for the required methodology first
        if method.lower() == "selectkbest":
            prettyPrint("Selecting %s best features from dataset" % targetDim, "debug")
            kBestSelector = SelectKBest(k=targetDim)
            X_new = kBestSelector.fit_transform(X, y).tolist()
        elif method.lower() == "pca":
            prettyPrint("Extracting %s features from dataset using PCA" % targetDim, "debug")
            pcaExtractor = PCA(n_components=targetDim)
            # Make sure vectors in X are positive
            X_new = pcaExtractor.fit_transform(X, y).tolist()
        else:
            prettyPrint("Unknown dimensionality reduction method \"%s\"" % method, "warning")
            return X

    except Exception as e:
        prettyPrint("Error encountered in \"reduceDimensionality\": %s" % e, "error")
        return X

    # Return the reduced dataset
    return X_new

def gatherStatsFromLog(fileName, expType, accuracyMode):
    """ Parses a classification dump file to calculate accuracies and confusion matrix """
    if not os.path.exists(fileName):
        prettyPrint("File \"%s\" does not exist. Exiting." % fileName, "warning")
        return False

    fileContent = open(fileName).read()
    # Group results by tree depth
    allLines = fileContent.split('\n')
    allDepths = {}
    currentDepth, currentTrends = "", []
    skip = True #TODO contributes to focusing on a certain depth/dimensionality
    print "[*] Parsing content..."
    lineCount = 0
    for line in allLines:
        if line.lower().find("tree depth:") != -1 or line.lower().find("target dimensionality:") != -1:
            # TODO: Focusing on the tree depth of 8 and the target dimensionality of 64
            if line.lower().find("tree depth: 8") == -1 and line.lower().find("target dimensionality: 64") == -1:
                prettyPrint("Skipping %s" % line, "debug")
                # Make sure we merge the 10th iteration
                if len(currentTrends) > 0:
                    if currentDepth in allDepths.keys():
                        prettyPrint("Merging trends at %s" % line, "debug")
                        allDepths[currentDepth] = mergeTrends(allDepths[currentDepth], currentTrends)
                        currentTrends = []
                skip = True
                continue
            skip = False
            currentDepth = line.split(": ")[1]
            prettyPrint("Processing %s:" % line, "debug")
            if len(currentTrends) > 0:
                # Store previous depth and reset it
                if currentDepth in allDepths.keys():
                    prettyPrint("Merging trends at %s" % line, "debug")
                    allDepths[currentDepth] = mergeTrends(allDepths[currentDepth], currentTrends)
                else:
                    prettyPrint("Adding new trend's list at %s" % line, "debug")
                    allDepths[currentDepth] = currentTrends
                currentTrends = []
        elif line.find("Class") != -1 and not skip:
            #lineCount += 1
            # Extract class and predicted
            if expType == "exp1":
                currentClass = line.split(',')[0].split(':')[1]
                currentPredicted = line.split(',')[1].split(':')[1]
            elif expType == "exp2":
                currentClass = line.split()[2][:-1]
                currentPredicted = line.split()[-1]
            else:
                prettyPrint("Unsupported experiment type \"%s\". Exiting" % expType, "debug")
            trend = "%s (as) %s" % (currentClass, currentPredicted)
            # Check whether trend exists in current trends
            trendIndex, oldTrend = findTrend(trend, currentTrends)
            if trendIndex != -1 and len(oldTrend) > 0:
                # If yes, update count
                newTrend = (trend, currentTrends[trendIndex][1]+1)
                #print newTrend
                currentTrends.pop(trendIndex)
                # Add to currentTrends
                currentTrends.append(newTrend)
            else:
                # else, add and set to zero 
                newTrend = (trend, 1)
                # Add to currentTrends
                currentTrends.append(newTrend)

    # Now sort the trends for all Depths
    prettyPrint("Sorting trends according to occurrence.")
    for tDepth in allDepths:
        allDepths[tDepth].sort(cmp=cmpTuple)

    # Display ordered trends
    keys = [int(x) for x in allDepths.keys()]
    keys.sort()
    allClasses, trends = [], []
    for tDepth in keys:
        print len(allDepths[str(tDepth)])
        print "================================"
        print "[*] Dimensionality /  Depth: %s" % tDepth
        print "================================"
        total = 0
        for trend in allDepths[str(tDepth)]:
            trends.append(trend)
            print "[*] %s encountered %s time(s)" % (trend[0], trend[1])
            total += trend[1]
            # Parse trend name
            class1 = trend[0].split(" (as) ")[0]
            class2 = trend[0].split(" (as) ")[1]
            if not class1 in allClasses:
                allClasses.append(class1)
            if not class2 in allClasses:
                allClasses.append(class2)

        # Sort classes alphabetically
        allClasses.sort()
        encodedClasses = {i+1: allClasses[i] for i in range(len(allClasses))}
        print "----------------------------------"
        print "[*] Total trend occurrences: %s" % total
        print "----------------------------------"
        # 2- Build a matrix of zeros
        confusionMatrix = numpy.zeros((len(allClasses), len(allClasses)))
        # 3- iterate over trends and extraxt classes
        for trend in trends:
            class1, class2 = trend[0].split(" (as) ")[0], trend[0].split(" (as) ")[1]
            count = trend[1]
# 4- Populate corresponding cells
            #print allClasses.index(class1), allClasses.index(class2), count
            confusionMatrix[allClasses.index(class1)][allClasses.index(class2)] += count
        # 5-Save to file 
        newFileName = fileName.replace("classificationlog", "confusionmatrix").replace(".txt", ".csv")
        numpy.savetxt(newFileName, confusionMatrix, delimiter=',', fmt='%i')
        # 6- Save class indices to file
        #f = open(fileName.replace("classificationlog", "confusionMatrix"), "a")
        correct = int(confusionMatrix.trace())
        # Update the count of correct instances according to the accuracy mode
        for trend in trends:
            original, classified = trend[0].split(" (as) ")[0], trend[0].split(" (as) ")[1]
            if accuracyMode == "viceversa":
                # Count (A+B) classified as (B+A) as correct.
                if original != classified and numOfMismatches(original, classified) <= 1:
                    correct += trend[1]
                    prettyPrint("Vice-versa trend %s found. Updating correctly-classified trends." % trend[0], "debug")
            elif accuracyMode == "jit":
                # Count (X+Jit) classified as (Jit) as correct ==> Jit is dominant
                if original != classified and original.find("Ji") != -1 and classified.find("Ji") != -1:
                    correct += trend[1]
                    prettyPrint("Jit trend %s found. Updating correctly-classified trends." % trend[0], "debug")
            elif accuracyMode == "both":
                # Implement both accuracy modes
                if original != classified and original.find("Ji") != -1 and classified.find("Ji") != -1:
                    correct += trend[1]
                    prettyPrint("Jit trend %s found. Updating correctly-classified trends." % trend[0], "debug")
                elif original != classified and numOfMismatches(original, classified) <= 1:
                    correct += trend[1]
                    prettyPrint("Vice-versa trend %s found. Updating correctly-classified trends." % trend[0], "debug")

        incorrect = int(total - correct)
        accuracy = round((float(correct)/float(total))*100.0, 2)
        #f.write("\n Correctly classified: %s, incorrectly classified: %s, Classification accuracy: %s%%, Total trends: %s\n" % (correct, incorrect, accuracy, total))
        #f.write("\n %s" % str(encodedClasses))
        #f.close()
        #print lineCount
        print "----------------------------------"
        print "[*] Accuracy mode: %s\n[*] Correctly classified: %s\n[*] Incorrectly classified: %s\n[*] Classification accuracy: %s%%\n[*] Total trends: %s" % (accuracyMode, correct, incorrect, accuracy, total)
        print "----------------------------------"
        allKeys = encodedClasses.keys()
        allKeys.sort()
        for k in allKeys:
            print "[%s] =>  \"%s\" => %s" % (k, chr(k+96).upper(), encodedClasses[k])    



