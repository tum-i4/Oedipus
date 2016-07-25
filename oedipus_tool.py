#!/usr/bin/python

###########################
# Oedipus Utility imports #
###########################
from Oedipus.utils.misc import *
from Oedipus.utils.graphics import *
from Oedipus.utils.data import *
###########################
# Oedipus Service imports #
###########################
from Oedipus.gadgets import classification
from Oedipus.gadgets import clustering
from Oedipus.gadgets import feature_extraction
from Oedipus.gadgets import data_visualization
from Oedipus.gadgets import program_generation
######################
# OS Utility imports #
######################
#from __future__ import division
import time, sys, os, subprocess
import shutil, glob, argparse, random
import numpy
from Levenshtein import distance


garbage = []

def defineArguments():
    parser = argparse.ArgumentParser(prog="oedipus_tool.py", description="Uses the \"Oedipus\" platform to learn and cluster/classify Tigress-generated obfuscated programs according to the transformations they employ.", usage="python oedipus_tool.py [options]")
    parser.add_argument("-m", "--mode", help="The requested mode of operation.", choices=["generate", "classify-exp1", "classify-exp2", "extract", "extract-from-traces", "visualize", "gather-stats", "filter-traces"], required=True, default="classify-exp1")
    parser.add_argument("-g", "--statlogfile", help="The file containing a dump of classification results.", required=False, default="log.txt")
    parser.add_argument("-s", "--sourcedir", help="The path to the directory containing the [un]obfuscated source code files.", required=False, default=".")
    parser.add_argument("-p", "--originalprograms", help="The path to the directory containing the original, unobfuscated programs.", required=False, default=".")
    parser.add_argument("-d", "--tigressdir", help="The path to the directory of \"tigress\".", required=False, default=".")
    parser.add_argument("-f", "--obfuscationfunction", help="The name of the function to obfuscate e.g. \"main\"", required=False, default="main")
    parser.add_argument("-r", "--filterfunction", help="The function to focus on during trace filteration.", required=False, default="main")
    parser.add_argument("-o", "--obfuscationlevel", help="The number of obfuscation combinations to employ.", required=False, default=1) 
    parser.add_argument("-a", "--algorithm", help="The machine learning algorithm to use.", required=False, default="tree", choices=["bayes", "tree"])
    parser.add_argument("-t", "--datatype", help="The format of data to consider e.g. numerical, traces, etc.", required=False, default="tfidf")
    parser.add_argument("-l", "--datalabel", help="The type of the label to load", required=False, default="label", choices=["label", "metadata"])
    parser.add_argument("-k", "--kfold", help="The number of folds to consider for K-fold cross validation.", required=False, default=10)
    parser.add_argument("-x", "--dimension", help="The dimension to which the data is projected before plotting.", required=False, default=2)
    parser.add_argument("-z", "--visualalgorithm", help="The algorithm used to project data into 2- or 3-dimensional space.", required=False, default="tsne", choices=["tsne", "pca"])
    parser.add_argument("-v", "--verbose", help="Displays debug messages on the screen.", default="no", choices=["yes", "no"], required=False)
    return parser

def main():
    try:
 
        argumentParser = defineArguments()
        arguments = argumentParser.parse_args()
        prettyPrint("Welcome to \"Oedipus\". Riddle me this!")

        #################################################
        # MODE 1: Generate obfuscated source code files #
        #################################################
        if arguments.mode == "generate":
           if arguments.verbose == "yes":
               prettyPrint("Generating obfusted programs for programs under \"%s\"" %  arguments.sourcedir, "debug")
           # Get programs from source directory [random/pre-existent]
           sourceFiles = sorted(glob.glob("%s/*.c" % arguments.sourcedir))
           if len(sourceFiles) < 1:
               prettyPrint("No files were found in \"%s\". Exiting" % arguments.sourcedir, "error")
               return

           generationStatus = program_generation.generateObfuscatedPrograms(sourceFiles, arguments.tigressdir, int(arguments.obfuscationlevel), arguments.obfuscationfunction) # Generate obfuscated programs
            
           prettyPrint("Successfully generated obfuscated programs")
        
        #########################################################
        # MODE 2: Extract features from obfuscated source files #
        #########################################################
        elif arguments.mode == "extract":
            # Load obfuscated files
            if not os.path.exists(arguments.sourcedir):
                prettyPrint("Unable to locate \"%s\". Exiting" % arguments.sourcedir, "error")
                return
            sourceFiles = sorted(glob.glob("%s/*.c" % arguments.sourcedir))
            if len(sourceFiles) < 1:
                prettyPrint("No files were found in \"%s\". Exiting" % arguments.sourcedir)
            
            # Remove source files without ".label" files
            for targetFile in sourceFiles:            
                if not os.path.exists(targetFile.replace(".c", ".label")):
                    prettyPrint("File \"%s\" does not have a label/metadata file. Removing" % targetFile, "warning")
                    sourceFiles.pop( sourceFiles.index(targetFile) )

            ########################################################################
            # (2.0) Extract TF-IDF features from GDB generated traces of KLEE inputs
            prettyPrint("Extracting TF-IDF from GDB traces")
            if not feature_extraction.extractTFIDF(arguments.sourcedir, sourceFiles):
                prettyPrint("Could not extract features from source files. Exiting", "error")
                return
            ########################################################################

            prettyPrint("Alright!! Alles in Ordnung.", "info2")
            cleanUp()
            return

        ###########################################################
        # MODE 3: Project data samples into <x>-dimensional space #
        ###########################################################
        elif arguments.mode.find("visualize") != -1:
            if arguments.mode == "visualize":
                prettyPrint("Plotting data into %s-dimensional space with \"%s\" features." % (arguments.dimension, arguments.datatype))
                data_visualization.visualizeData(arguments.sourcedir, arguments.datatype, arguments.dimension, algorithm=arguments.visualalgorithm)
            else:
                data_visualization.visualizeOriginal(arguments.sourcedir, arguments.datatype, arguments.dimension, algorithm=arguments.visualalgorithm)

        ##############################################################################
        # MODE 4: Classify obfuscated programs using knowledge-based  classification #
        ##############################################################################
        elif arguments.mode == "classify-exp1":
           # Check the requested algorithm
           if arguments.algorithm == "bayes":
               # Classify using Naive Bayes
               if arguments.datatype.find("idf") == -1:
                   prettyPrint("Naive Bayes does not support the data type \"%s\". Exiting" % arguments.datatype, "warning")
                   #return
               # Load data from source directory
               X, y, allClasses = loadFeaturesFromDir(arguments.sourcedir, arguments.datatype, arguments.datalabel)
               reductionMethod = raw_input("Please choose a dimensionality reduction method (selectkbest/pca): ").lower()
               classificationLog = open("classificationlog_%s_exp1_%s_%s.txt" % (arguments.datatype, reductionMethod, arguments.algorithm), "a") # A file to log all classification labels
               classificationLog.write("Experiment 1 - Algorithm: %s, Datatype: %s\n" % (arguments.algorithm, arguments.datatype))
               if reductionMethod == "selectkbest":
                   accuracies, timings = [], []
                   targetDimensions = [8, 16, 32, 64, 128]#[64, 128, 256, 512, 1000]
                   for dimension in targetDimensions:
                       if arguments.verbose == "yes":
                           prettyPrint("Training a naive Bayes classifier with %s selected \"%s\" features" % (dimension, arguments.datatype), "debug")
                       accuracyRates, allProbabilities, allTimings, groundTruthLabels, predictedLabels = classification.classifyNaiveBayesKFold(X, y, kFold=int(arguments.kfold), reduceDim=reductionMethod, targetDim=dimension)
                       prettyPrint("Average classification accuracy: %s%%" % (averageList(accuracyRates)*100.0), "output")
                       accuracies.append(averageList(accuracyRates))
                       timings.append(averageList(allTimings))
                       # Log classifications
                       for foldIndex in range(len(predictedLabels)):
                           classificationLog.write("Target Dimensionality: %s\n" % dimension)
                           for labelIndex in range(len(predictedLabels[foldIndex])):
                               classificationLog.write("Class:%s,Predicted:%s\n" % (allClasses[groundTruthLabels[foldIndex][labelIndex]], allClasses[predictedLabels[foldIndex][labelIndex]]))
                   
                   classificationLog.close()
                   # Plot accuracies graph
                   prettyPrint("Plotting accuracies")
                   data_visualization.plotAccuracyGraph(targetDimensions, accuracies, "Number of Selected Features", "Classification Accuracy", "Classification Accuracy: Selected Features (%s)" % arguments.datatype, "accuracy_%s_exp1_%s_selectkbest.pdf" % (arguments.datatype, arguments.algorithm)) 
                   # Plot performance graph
                   print timings
                   #prettyPrint("Plotting performance")
                   #data_visualization.plotAccuracyGraph(targetDimensions, timings, "Number of Selected Features", "Classification Timing (sec)", "Classification Timing: Selected Features (%s)" % arguments.datatype) 
                  
               elif reductionMethod == "pca":
                   accuracies, timings = [], []
                   targetDimensions = [8, 16, 32, 64, 128]#[2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]
                   for dimension in targetDimensions:
                       if arguments.verbose == "yes":
                           prettyPrint("Training a naive Bayes classifier with %s extracted \"%s\" features" % (dimension, arguments.datatype), "debug")
                       accuracyRates, allProbabilities, allTimings, groundTruthLabels, predictedLabels = classification.classifyNaiveBayesKFold(X, y, kFold=int(arguments.kfold), reduceDim=reductionMethod, targetDim=dimension)
                       prettyPrint("Average classification accuracy: %s%%" % (averageList(accuracyRates)*100.0), "output")
                       accuracies.append(averageList(accuracyRates))
                       timings.append(averageList(allTimings))
                       # Log classifications
                       for foldIndex in range(len(predictedLabels)):
                           classificationLog.write("Target Dimensionality: %s\n" % dimension)
                           for labelIndex in range(len(predictedLabels[foldIndex])):
                               classificationLog.write("Class:%s,Predicted:%s\n" % (allClasses[groundTruthLabels[foldIndex][labelIndex]], allClasses[predictedLabels[foldIndex][labelIndex]]))

                   classificationLog.close()
                   # Plot accuracies graph
                   prettyPrint("Plotting accuracies")
                   data_visualization.plotAccuracyGraph(targetDimensions, accuracies, "Number of Extracted Features", "Classification Accuracy", "Classification Accuracy: PCA (%s)" % arguments.datatype, "accuracy_%s_exp1_%s_pca.pdf" % (arguments.datatype, arguments.algorithm))
                   # Plot performance graph
                   print timings
                   #prettyPrint("Plotting performance")
                   #data_visualization.plotAccuracyGraph(targetDimensions, timings, "Number of Extracted Features", "Classification Timing (sec)", "Classification Timing: PCA (%s)" % arguments.datatype)

               else:    
                   accuracyRates, allProbabilities, allTimings, predictedLabels = classification.classifyNaiveBayes(X, y, kFold=int(arguments.kfold))
                   prettyPrint("Average classification accuracy: %s%%, achieved in an average of %s seconds" % (averageList(accuracyRates)*100.0, averageList(allTimings)), "output")
           ####################
           # Using CART trees #
           ####################
           elif arguments.algorithm == "tree":
               # Classify using CART trees
               if arguments.datatype != "triton":
                   prettyPrint("It is recommended to use \".triton\" features", "warning")
               # Load data from source directory
               X, y, allClasses = loadFeaturesFromDir(arguments.sourcedir, arguments.datatype, arguments.datalabel)
               splittingCriterion = raw_input("Please choose a splitting criterion (gini/entropy): ")
               classificationLog = open("classificationlog_%s_exp1_%s_%s.txt" % (arguments.datatype, splittingCriterion, arguments.algorithm), "a") # A file to log all classification labels
               classificationLog.write("Experiment 1 - Algorithm: %s, Datatype: %s\n" % (arguments.algorithm, arguments.datatype))
               #maxDepth = raw_input("Please choose a maximum depth for the tree (0 = Maximum Possible): ") # Should be (2,4,8,16)
               accuracies, timings, allDepths = [], [], [2,3,4,5,6,7,8,10,12,14,16]#,32,64]
               for maxDepth in allDepths:
                   if arguments.verbose == "yes":
                       prettyPrint("Training a \"CART\" with \"%s\" criterion and maximum depth of %s" % (splittingCriterion, maxDepth), "debug")
                   accuracyRates, allProbabilities, allTimings, groundTruthLabels, predictedLabels = classification.classifyTreeKFold(X, y, int(arguments.kfold), splittingCriterion, int(maxDepth), visualizeTree=False)
                   #print accuracyRates, allProbabilities
                   prettyPrint("Average classification accuracy: %s%%" % (averageList(accuracyRates)*100.0), "output")
                   accuracies.append(averageList(accuracyRates))
                   timings.append(averageList(allTimings))
                   # Log classifications
                   for foldIndex in range(len(predictedLabels)):
                       classificationLog.write("Tree Depth: %s\n" % maxDepth)
                       for labelIndex in range(len(predictedLabels[foldIndex])):
                           classificationLog.write("Class:%s,Predicted:%s\n" % (allClasses[groundTruthLabels[foldIndex][labelIndex]], allClasses[predictedLabels[foldIndex][labelIndex]]))

               classificationLog.close()
               # Plot accuracies graph
               prettyPrint("Plotting accuracies for \"%s\" criterion" % splittingCriterion)
               data_visualization.plotAccuracyGraph(allDepths, accuracies, "Maximum Tree Depth", "Classification Accuracy", "Classification Accuracy: %s (%s)" % (splittingCriterion, arguments.datatype), "accuracy_%s_exp1_%s_%s.pdf" % (arguments.datatype, splittingCriterion, arguments.algorithm))
               # Plot performance graph
               #prettyPrint("Plotting timings")
               #data_visualization.plotAccuracyGraph(allDepths, timings, "Maximum Tree Depth", "Classification Timing (sec)", "Classification Timing: %s (%s)" % (splittingCriterion, arguments.datatype))
               print timings
 
           return

        ##################################################################
        # MODE 6: Classify obfuscated programs using the 36-4 experiment #
        ##################################################################
        elif arguments.mode == "classify-exp2":
            # Retrieve the list of all programs
            allPrograms = glob.glob("%s/*.c" % arguments.originalprograms)#list(set(sorted(glob.glob("%s/*.c" % arguments.sourcedir))) - set(sorted(glob.glob("%s/*-*.c" % arguments.sourcedir))))
            allPrograms.sort() # Makes it easier to keep track of current programs in batch
            totalPrograms = len(allPrograms)
            prettyPrint("Successfully retrieved %s original programs" % totalPrograms)
            chunkSize =  totalPrograms/int(arguments.kfold) # 4 = 40 / 10 (default)
            if arguments.algorithm == "tree":
                criterion = raw_input("Please choose a splitting criterion (gini/entropy): ")
                allValues = [2,3,4,5,6,7,8,10,12,14,16]#,32,64] # The allowed depths of the tree
            elif arguments.algorithm == "bayes":
                criterion = raw_input("Please choose a dimensionality reduction method (SelectKBest/PCA): ").lower()
                allValues = [8,16,32,64,128]# if criterion.lower() == "selectkbest" else [8,16,32,64,128]       
            # Define the structure of the accuracy and timing matrices
            allAccuracyRates, allTimings = numpy.zeros((int(arguments.kfold), len(allValues))), numpy.zeros((int(arguments.kfold), len(allValues)))
            classificationLog = open("classificationlog_%s_exp2_%s_%s.txt" % (arguments.datatype, criterion, arguments.algorithm), "a") # A file to log all classification labels
            classificationLog.write("Experiment 2 - Algorithm: %s, Datatype: %s\n" % (arguments.algorithm, arguments.datatype))
            for currentCycle in range(10):
                prettyPrint("Cycle #%s out of %s cycles" % (currentCycle+1, int(arguments.kfold)))
                trainingPrograms, testPrograms = [] + allPrograms, []
                # Specify the indices of the training and test datasets
                testStartIndex = (totalPrograms + (chunkSize * currentCycle)) % totalPrograms
                testStopIndex = testStartIndex + chunkSize
                if arguments.verbose == "yes":
                    prettyPrint("Retrieving training and test programs for the current cycle", "debug")
                # Populate the test dataset
                testPrograms = trainingPrograms[testStartIndex:testStopIndex]
                # Remove the indices from trainingPrograms
                trainingPrograms = [x for x in trainingPrograms if not x in trainingPrograms[testStartIndex:testStopIndex]]
                if arguments.verbose == "yes":
  		    prettyPrint("Original training programs: %s, original test programs: %s" % (len(trainingPrograms), len(testPrograms)), "debug")
                # Now load the training and test samples from the source directory
                # 1- First we need to retrieve the obfuscated versions of the
                tempTraining, tempTest = [], []
                for program in trainingPrograms:
                    programName = program.replace(arguments.originalprograms, "").replace("/","") # Isolate program name
                    # TODO: Important: For 40 programs, programs are like "anagram_1231231231_12.c"
                    # TODO: for "obf" programs, programs are like "empty-Seed1-Random......-addOpaque16.c"
                    separator = "_" if arguments.sourcedir.find("40programs") != - 1 else "-"
                    #print "%s/%s%s*.%s" % (arguments.sourcedir, programName.replace(".c", ""), separator, arguments.datatype)
                    obfuscatedVersions = glob.glob("%s/%s%s*.%s" % (arguments.sourcedir, programName.replace(".c", ""), separator, arguments.datatype)) 
                    #print programName, len(obfuscatedVersions)
                    #print "%s/%s_*.%s" % (arguments.sourcedir, programName.replace(".c", ""), arguments.datatype)
                    if len(obfuscatedVersions) > 0:
                        tempTraining += obfuscatedVersions
                    #print programName, len(obfuscatedVersions)
                for program in testPrograms:
                    programName = program.replace(arguments.originalprograms, "").replace("/","") # Isolate program name
                    # TODO: Important: For 40 programs, programs are like "anagram_1231231231_12.c"
                    # TODO: for "obf" programs, programs are like "empty-Seed1-Random......-addOpaque16.c"
                    separator = "_" if arguments.sourcedir.find("40programs") != - 1 else "-"
                    obfuscatedVersions = glob.glob("%s/%s%s*.%s" % (arguments.sourcedir, programName.replace(".c", ""), separator, arguments.datatype)) 
                    if len(obfuscatedVersions) > 0:
                       tempTest += obfuscatedVersions
                trainingPrograms, testPrograms = tempTraining, tempTest # Update the training and test programs
                if arguments.verbose == "yes":
                    prettyPrint("Successfully retrieved %s training and %s test programs" % (len(trainingPrograms), len(testPrograms)), "debug")
                # (Added January 15): Generate the TF-IDF features on the fly
                if arguments.verbose == "yes":
                    prettyPrint("Generating TF-IDF features for the current training and test traces", "debug")
                if feature_extraction.extractTFIDFMemoryFriendly(trainingPrograms, arguments.datatype, 128, "%s_tr" % arguments.datatype):
                    prettyPrint("Successfully generated TF-IDF features for the current training batch") 
                else:
                    prettyPrint("Unable to generate TF-IDF features for the current training batch", "warning")
                    continue
                # Now for the test batch
                if feature_extraction.extractTFIDFMemoryFriendly(testPrograms, arguments.datatype, 128, "%s_te" % arguments.datatype):
                    prettyPrint("Successfully generated TF-IDF features for the current test batch")
                else:
                    prettyPrint("Unable to generate TF-IDF features for the current test batch", "warning")
                    continue

                # Now load the programs of the given datatype
                prettyPrint("Loading training and test instances")
                Xtr, ytr, allClassestr = loadFeaturesFromList(trainingPrograms, "%s_tr" % arguments.datatype, arguments.datalabel)
                Xte, yte, allClasseste = loadFeaturesFromList(testPrograms, "%s_te" % arguments.datatype, arguments.datalabel, allClassestr)
                # Now apply the classification algorithm 
                for value in allValues:
                    ##############
                    # CART Trees #
                    ##############
                    if arguments.algorithm == "tree":
                        prettyPrint("Training a \"CART\" with \"%s\" criterion and maximum depth of %s" % (criterion, value), "debug")
                        currentAccuracyRate, currentTiming, currentProbabilities, predictedLabels = classification.classifyTree(Xtr, ytr, Xte, yte, criterion, int(value), visualizeTree=False)
                        prettyPrint("Classification accuracy with \"%s\" and \"%s\" is: %s%%" % (criterion, value, (currentAccuracyRate*100.0)), "output")
                        #print "before!!!! currentCycle: %s, value: %s, allValues.index(value): %s" % (currentCycle, value, allValues.index(value))
                        allAccuracyRates[currentCycle][allValues.index(value)] = currentAccuracyRate
                        allTimings[currentCycle][allValues.index(value)] = currentTiming
                        #print "after assignments"
                        # Log the results
                        classificationLog.write("Depth: %s\n" % value)
                        #print len(yte), len(predictedLabels), len(testPrograms)
                        for index in range(len(testPrograms)): 
                            classificationLog.write("%s: Class: %s, Predicted: %s\n" % (testPrograms[index], allClasseste[yte[index]], allClasseste[predictedLabels[index]]))
                        #print "after writing"
                    ###########################
                    # Multinomial Naive Bayes #
                    ###########################
                    elif arguments.algorithm == "bayes":
                        prettyPrint("Training a \"Multinomial Naive Bayes\" with \"%s\" criterion and dimensionality of %s" % (criterion, value), "debug")
                        currentAccuracyRate, currentTiming, currentProbabilities, predictedLabels = classification.classifyNaiveBayes(Xtr, ytr, Xte, yte, criterion, int(value))

                        #print accuracyRates, allProbabilities
                        prettyPrint("Classification accuracy with \"%s\" and \"%s\" is: %s%%" % (criterion, value, (currentAccuracyRate*100.0)), "output")
                        allAccuracyRates[currentCycle][allValues.index(value)] = currentAccuracyRate
                        allTimings[currentCycle][allValues.index(value)] = currentTiming
                        # Log the results
                        classificationLog.write("Dimensionality: %s\n" % value)
                        #print len(yte), len(predictedLabels), len(testPrograms)
                        for index in range(len(testPrograms)): 
                            classificationLog.write("%s: Class: %s, Predicted: %s\n" % (testPrograms[index], allClasseste[yte[index]], allClasseste[predictedLabels[index]]))
                
                # TODO (Added January 15): Remove all TF-IDF files of the current batch
                if arguments.verbose == "yes":
                    prettyPrint("Removing all TF-IDF files of the current batch", "debug")
                rmCounter = 0
                for featureFile in glob.glob("%s/*.%s_t*" % (arguments.sourcedir, arguments.datatype)): # TODO: This will remove tfidf_both you stupid fuck!!
                    os.unlink(featureFile)
                    rmCounter += 1
                prettyPrint("Successfully removed %s files" % rmCounter)
                    
            classificationLog.close()
            # Now average the scored results stored in the matrices
            pointsX, pointsYacc, pointsYtime = [], [], []
            for value in allValues:
                pointsX.append(value)
                pointsYacc.append(averageList(allAccuracyRates[:,allValues.index(value)]))
                pointsYtime.append(averageList(allTimings[:,allValues.index(value)]))
             # Plot accuracies and timings graphs
            if arguments.algorithm == "tree":
                xAxisLabel = "Maximum Tree Depth"
            elif arguments.algorithm == "bayes":
                xAxisLabel = "Selected Features" if criterion == "select" else "Extracted Features"
           
            prettyPrint("Plotting accuracies for \"%s\" criterion" % criterion)
            data_visualization.plotAccuracyGraph(pointsX, pointsYacc, xAxisLabel, "Classification Accuracy", "Classification Accuracy: %s (%s)" % (criterion, arguments.datatype), "accuracy_%s_exp2_%s_%s.pdf" % (arguments.datatype, criterion, arguments.algorithm))
            #prettyPrint("Plotting timings")
            #data_visualization.plotAccuracyGraph(pointsX, pointsYtime, "Maximum Tree Depth", "Classification Timing (sec)", "Classification Timing: %s (%s)" % (criterion, arguments.datatype))

        ####################################
        # MODE X : Filter generated traces #
        ####################################
        elif arguments.mode == "filter-traces":
            # Retrieve the necessary parameters
            inExtension = raw_input("Input extension (Default: dyndis): ")
            outExtension = raw_input("Output extension (Default: dyndis_raw): ")
            filterMode = raw_input("Filteration mode {raw (Default), mem, both}: ")
            if filterTraces(arguments.sourcedir, inExtension, filterMode, outExtension, arguments.filterfunction):
                prettyPrint("Successfully filtered \"%s\" traces to \"%s\" traces using the \"%s\" filter" % (inExtension, outExtension, filterMode))
            else:
               prettyPrint("Some error occurred during filteration", "warning")

        ########################################################
        # MODE XI: Generate TF-IDF feature vectors from traces #
        ########################################################
        elif arguments.mode == "extract-from-traces":
            # Retrieve the necessary paramters
            inExtension = raw_input("Input extension (Default: dyndis): ")
            outExtension = raw_input("Output extension (Default: tfidf_raw): ")
            maxFeatures = int(raw_input("Maximum features: "))
            if feature_extraction.extractTFIDFMemoryFriendly(arguments.sourcedir, inExtension, maxFeatures, outExtension):
                prettyPrint("Successfully extracted %s TF-IDF features from traces with \"%s\" extension" % (maxFeatures, inExtension))
            else:
                prettyPrint("Some error occurred during TF-IDF feature extraction", "warning")

    except Exception as e:
        #global garbage
        prettyPrint("Error encountered in \"main\": %s at line %s" % (e, sys.exc_info()[2].tb_lineno), "error")
        #print garbage
        cleanUp()
        return

if __name__ == "__main__":
    main()


