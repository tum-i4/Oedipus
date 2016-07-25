#!/usr/bin/python

###################
# Library imports #
###################
from Oedipus.utils.data import *
from Oedipus.utils.misc import *
from Oedipus.utils.graphics import *
import glob, subprocess, time, os, threading, shutil
import numpy
import ghmm
from gensim import corpora, models, similarities
from gensim.corpora import dictionary
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
#############################
# Defining Global Variables #
#############################
currentThreads = 0 # Quick, nasty way to keep track of the number of currently-running threads
####################
# Defining Methods #
####################
def generateObjdumpDisassembly(outFile, inExt=".out", outExt=".objdump"):
    """ Generates an Objdump of an executable """
    # Check whether file is executable using "file"
    fileOutput = subprocess.Popen(["file", outFile], stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
    if fileOutput.lower().find("executable") == -1:
        prettyPrint("The file \"%s\" is not an executable" % outFile, "warning")
        return False
    # Generate the objdump disassembly 
    objdumpFile = open(outFile.replace(inExt, outExt), "w")
    objdumpArgs = ["objdump", "--disassemble", outFile]
    objdumpOutput = subprocess.Popen(objdumpArgs, stderr=subprocess.STDOUT, stdout=objdumpFile).communicate()[0]
    # Check if the file has been generated and not empty
    if not os.path.exists(outFile.replace(inExt, outExt)) or os.path.getsize(outFile.replace(inExt, outExt)) < 1:
        prettyPrint("Could not find a (non-empty) objdump disassembly file for \"%s\"" % outFile, "warning")
        return False

    return True

def generateGDBScript(logFile="gdb.txt", inputFile="", runArgs=[]):
    """ Generates the GDB script needed for trace generation """
    gdbScript = open(logFile.replace(".txt",".script_%s" % str(time.time())[:-3]), "w")
    if len(runArgs) > 0 or len(inputFile) > 0:
        # Bundle all the arguments in one string
        args = " ".join([a for a in runArgs])
        # Beware of the "step" versus "stepi"
        gdbScript.write("set logging file %s\nset logging on\nset height 0\nset $_exitcode = -999\nset $_instructioncount = 0\nbreak __libc_start_main\nrun `< %s`\nwhile $_instructioncount <= 50000\n\tx/i $pc\n\tstepi\n\tif $_exitcode != -999\n\t\tset logging off\n\t\tquit\n\tend\n\tset $_instructioncount = $_instructioncount + 1\nend\nset logging off\nquit" % (logFile, inputFile))
    else:
        gdbScript.write("set logging file %s\nset logging on\nset height 0\nset $_exitcode = -999\nset $_instructioncount = 0\nbreak __libc_start_main\nrun\nwhile $_instructioncount <= 50000\n\tx/i $pc\n\tstepi\n\tif $_exitcode != -999\n\t\tset logging off\n\t\tquit\n\tend\n\tset $_instructioncount = $_instructioncount + 1\nend\nset logging off\nquit" % logFile)
    gdbScript.close()
    return gdbScript.name

def compileFile(targetFile):
    """ Compiles a source files for feature extraction """
    aoutTimestamp = str(time.time()).replace(".","_")
    outFile = targetFile[targetFile.rfind("/")+1:].replace(".c",".out")
    outFile_strip = targetFile[targetFile.rfind("/")+1:].replace(".c", ".outs")
    #outFile = "%s_%s.out" % (fileName, aoutTimestamp)
    gccArgs = ["gcc", "-Wl,--unresolved-symbols=ignore-in-object-files","-std=c99", targetFile, "-o", outFile]
    gccArgs_strip = ["gcc", "-s", "-Wl,--unresolved-symbols=ignore-in-object-files","-std=c99", targetFile, "-o", outFile_strip]
    prettyPrint("Compiling \"%s\"" % targetFile, "debug")
    subprocess.Popen(gccArgs, stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
    prettyPrint("Compiling \"%s\" with \"-s\"" % targetFile, "debug")
    subprocess.Popen(gccArgs_strip, stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
    # Check if compilation succeeded by checking for existence of "a.out"
    if not os.path.exists(outFile) or not os.path.exists(outFile_strip):
        prettyPrint("Compiling \"%s\" failed. Skipping file" % targetFile , "warning")
        return "", ""
        
    return outFile, outFile_strip

def extractFeaturesFromITrace(instructionTrace):
    # Extracts numeric features from an instruction trace 
    # ... and returns a vector of such features
    instructionFeatures = []
    numOfArithmeticOps, numOfLogicOps, numOfJmpOps, numOfMovOps, numOfCallOps, totalNumOfOps = 0, 0, 0, 0, 0, 0
    instructionChangeFreq, sameAddressJmpFreq = 0, 0
    arithmeticToArithmetic, arithmeticToControl, arithmeticToLogic, arithmeticToData = 0, 0, 0, 0
    logicToLogic, logicToControl, logicToArithmetic, logicToData = 0, 0, 0, 0
    controlToControl, controlToArithmetic, controlToLogic, controlToData = 0, 0, 0, 0
    dataToData, dataToControl, dataToArithmetic, dataToLogic = 0, 0, 0, 0
    previousInstruction, visitedAddresses = "", []
    # Define instruction categories
    arithmeticInstructions = ["add", "sub", "mul", "sh", "rot", "inc", "dec", "shr", "shl", "addq"]
    logicInstructions = ["and", "or", "xor", "neg", "test", "cmp", "testb", "cmpq", "testl"]
    controlInstructions = ["jmp", "jne", "je", "jz", "jnz", "jg", "jge", "jl", "jle", "ja", "jae", "jb", "jbe", "jo", "jno", "jc", "jnc", "js", "jns", "loop", "call", "rep"]
    dataInstructions = ["mov", "loads", "stos", "pop", "push", "movzbl", "cmove", "movl", "movq"]
    for instruction in instructionTrace:
        # 1. Update the counters
        if instruction[0].find("add") != -1 or instruction[0].find("sub") != -1 or instruction[0].find("sh") != -1 or instruction[0].find("rot") != -1:
            numOfArithmeticOps += 1
        elif instruction[0].find("and") != -1 or instruction[0].find("or") != -1 or instruction[0].find("xor") != -1 or instruction[0].find("neg") != -1:
            numOfLogicOps += 1
        elif instruction[0].find("j") != -1:
            numOfJmpOps += 1
            if instruction[1] not in visitedAddresses:
                visitedAddresses.append(instruction[1]) # Append address to previously-visited addresses
            else:
                sameAddressJmpFreq += 1
        elif instruction[0].find("mov") != -1:
            numOfMovOps += 1
        elif instruction[0].find("call") != -1:
            numOfCallOps += 1
        totalNumOfOps += 1
        # 2. Update the instruction change frequency counters
        if previousInstruction != instruction[0]:
            instructionChangeFreq += 1
            # Check the category change
            if previousInstruction in arithmeticInstructions and instruction[0] in logicInstructions:
                arithmeticToLogic += 1
            elif previousInstruction in arithmeticInstructions and instruction[0] in controlInstructions:
                arithmeticToControl += 1
            elif previousInstruction in arithmeticInstructions and instruction[0] in dataInstructions:
                arithmeticToData += 1
            elif previousInstruction in arithmeticInstructions and instruction[0] in arithmeticInstructions:
                arithmeticToArithmetic += 1
            elif previousInstruction in logicInstructions and instruction[0] in logicInstructions:
                logicToArithmetic += 1
            elif previousInstruction in logicInstructions and instruction[0] in controlInstructions:
                logicToControl += 1
            elif previousInstruction in logicInstructions and instruction[0] in dataInstructions:
                logicToData += 1
            elif previousInstruction in controlInstructions and instruction[0] in controlInstructions:
                controlToControl += 1
            elif previousInstruction in controlInstructions and instruction[0] in arithmeticInstructions:
                controlToArithmetic += 1
            elif previousInstruction in controlInstructions and instruction[0] in logicInstructions:
                controlToLogic += 1
            elif previousInstruction in controlInstructions and instruction[0] in dataInstructions:
                controlToData += 1
            elif previousInstruction in dataInstructions and instruction[0] in dataInstructions:
                dataToData += 1
            elif previousInstruction in dataInstructions and instruction[0] in arithmeticInstructions:
                dataToArithmetic += 1
            elif previousInstruction in dataInstructions and instruction[0] in controlInstructions:
                dataToControl += 1
            elif previousInstruction in dataInstructions and instruction[0] in logicInstructions:
                dataToLogic += 1
            previousInstruction = instruction[0] # Update the previous instruction

    # Append all the features to the syscallFeatures
    instructionFeatures = [numOfArithmeticOps, numOfJmpOps, numOfMovOps, numOfCallOps, totalNumOfOps, instructionChangeFreq, sameAddressJmpFreq]
    instructionFeatures += [arithmeticToControl, arithmeticToLogic, arithmeticToData, controlToArithmetic, controlToLogic, controlToData]
    instructionFeatures += [dataToArithmetic, dataToLogic, dataToControl, logicToArithmetic, logicToControl, logicToData]

    return instructionFeatures

def innerListLevenshtein(l):
    # Calculate the average Levenshtein distance between strings within a list
    if len(l) < 2:
        #print "[*] List is too short. %s" % getTimestamp()
        return 0.0
    allDistances, index, ops = [], 1, 0
    for referenceAction in range(len(l)):
        for variableAction in range(index, len(l)):
            allDistances.append(float(distance(l[referenceAction], l[variableAction])))
            ops += 1
        index += 1
    return round(reduce(lambda x, y: x + y, allDistances) / ops, 2)

def _generateDisassembly(targetFile, outFile, outFile_s):
    """ Generates a ".dyndis" file for a given file using a randomly generated testcase """
    try:
        disassemblyFile = targetFile.replace(".c", ".dyndis")
        disassemblyFile_strip = targetFile.replace(".c", ".dyndiss")
        testCaseFile = targetFile.replace(".c",".input")
        testCase = getRandomNumber(5)
        open(testCaseFile, "w").write(testCase) # Write the test case to a ".input" file.
        if os.path.exists(disassemblyFile):
            prettyPrint("Disassembly file \"%s\" already exists. Skipping" % disassemblyFile, "warning")
            return True

        script = generateGDBScript(outFile.replace(".out", ".txt"), inputFile=testCaseFile)
        script_s = generateGDBScript(outFile_s.replace(".outs", ".txts"), inputFile=testCaseFile)
        # Launch the GDB script
        prettyPrint("Launching the GDB script. Release the Kraken!!")
        gdbOutput = subprocess.Popen(["gdb", "-batch", "-x", script, outFile], stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
        gdbOutput_s = subprocess.Popen(["gdb", "-batch", "-x", script_s, outFile_s], stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
        # Check that the output does not indicate erroneous runtime behavior
        if not checkTestCaseSuccess(gdbOutput) or not checkTestCaseSuccess(gdbOutput_s):
            prettyPrint("The test case \"%s\" crashed the file \"%s\". Skipping" % (testCase, targetFile), "warning")
            return False

        # Get the instruction trace of the process from "gdb.txt" and extract features from it
        if os.path.exists(outFile.replace(".out",".txt")):
            # Store the contents of "gdb.txt" as disassembly for further processing
            prettyPrint("Dumping dynamic disassembly to \"%s\" and \"%s\"" % (disassemblyFile, disassemblyFile_strip), "debug")
            gdbFile = open(disassemblyFile, "w")
            gdbFile_s = open(disassemblyFile_strip, "w")
            gdbFileContent = open(outFile.replace(".out",".txt")).read()
            gdbFileContent_s = open(outFile_s.replace(".outs", ".txts")).read()
            if gdbFileContent.find("Segmentation fault") != -1 or gdbFileContent_s.find("Segmentation Fault") != -1:
                prettyPrint("Test case \"%s\"crashed the file \"%s\". Skipping" % (testCase, targetFile), "warning")
                return False
            # Otherwise write content to file
            gdbFile.write(gdbFileContent)
            gdbFile.close()
            gdbFile_s.write(gdbFileContent_s)
            gdbFile_s.close()

        # Remove .txt files and .script files
        allfiles = glob.glob("./*.out*") + glob.glob("./*.script*") + glob.glob("./*.txt*") + glob.glob("./*.objdump*")
        for f in allfiles:
            os.unlink(f)
 
    except Exception as e:
        prettyPrint("Error encountered in \"_generateDisassembly\": %s" % e, "error")
        return False

    return True

def _generateDisassemblyFiles(targetFile, outFile, fileTestCases):
    """ Generates ".dyndis" files for the given file using KLEE testcases. Used for parallelization """
    try:
        for testCase in fileTestCases:
            # Check if disassembly file exists before running
            disassemblyFile = "%s_%s.dyndis" % (targetFile.replace(".c",""), testCase[testCase.rfind("/")+1:].replace(".txt",""))
            if os.path.exists(disassemblyFile):
                prettyPrint("Disassembly file \"%s\" already exists. Skipping" % disassemblyFile, "warning")
                continue
            # (2.b.i) Parse the KLEE test file and retrieve the list of arguments
            runArgs, inputFile = loadArgumentsFromKLEE(testCase)
            # (2.b.ii) Generate a GDB script to "run" with these two inputs
            generateGDBScript(outFile.replace(".out", ".txt"), inputFile=testCase.replace(".txt",".input"))
            # (2.b.iii) Launch the GDB script
            prettyPrint("Launching the GDB script. Release the Kraken!!")
            gdbOutput = subprocess.Popen(["gdb", "-batch", "-x", outFile.replace(".out",".script"), outFile], stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
            # Check that the output does not indicate erroneous runtime behavior
            if not checkTestCaseSuccess(gdbOutput):
                prettyPrint("The test case \"%s\" crashed the file \"%s\". Skipping" % (testCase, targetFile), "warning")
                continue
            # (2.b.iv) Get the instruction trace of the process from "gdb.txt" and extract features from it
            if os.path.exists(outFile.replace(".out",".txt")):
                # Store the contents of "gdb.txt" as disassembly for further processing
                prettyPrint("Dumping dynamic disassembly to \"%s\"" % disassemblyFile, "debug")
                gdbFile = open(disassemblyFile, "w")
                gdbFileContent = open(outFile.replace(".out",".txt")).read()
                if gdbFileContent.find("Segmentation fault") != -1:
                    prettyPrint("Test case \"%s\"crashed the file \"%s\". Skipping" % (testCase, targetFile), "warning")
                    continue
                # Otherwise write content to file
                gdbFile.write(gdbFileContent)
                gdbFile.close()
                # Also generate a label file for ease of retrieval
                labelFile = open(disassemblyFile.replace(".dyndis", ".label"), "w")
                labelFile.write("%s\n" % loadLabelFromFile(targetFile.replace(".c",".metadata"))[0])
                labelFile.close()
                  
            os.unlink(outFile.replace(".out",".txt")) # Remove the gdb logging file to avoid appending to it
            os.unlink(outFile.replace(".out",".script"))
        os.unlink(outFile)

    except Exception as e:
        prettyPrint("Error encountered in \"_generateDisassemblyFiles\": %s" % e, "error")
        return False

    return True

def extractTFIDF(sourceDir, sourceFiles):
    """ Extract TF-IDF features from GDB traces """
    try:
        for targetFile in sourceFiles:
            # (1) Compile the file
            outFile, outFile_strip = compileFile(targetFile)
            if outFile == "" or outFile_strip == "":
                prettyPrint("Unable to compile \"%s\". Skipping" % targetFile, "warning")
                continue
            # (1.a) Generate objdumps from binaries
            if generateObjdumpDisassembly(outFile) and generateObjdumpDisassembly(outFile_strip, ".outs", ".objdumps"):
                prettyPrint("%s.objdump and %s.objdumps have been successfully generated" % (outFile, outFile_strip))
                shutil.copy(outFile.replace(".out", ".objdump"), sourceDir)
                shutil.copy(outFile_strip.replace(".outs", ".objdumps"), sourceDir)

            # (2) Generate a disassembly trace file
            _generateDisassembly(targetFile, outFile, outFile_strip)

        # (3) After all files are done, load all the "dyndis" files and extract TF-IDF features from them.
        #disassemblyFiles = glob.glob("%s/*.dyndis" % sourceDir)
        #if len(disassemblyFiles) < 1:
        #    prettyPrint("Unable to retrieve \".dyndis\" files from \"%s\"" % sourceDir, "warning")
        #    return False
        #prettyPrint("Successfully retrieved %s \".dyndis\" files from \"%s\"" % (len(disassemblyFiles), sourceDir))
        #if extractTFIDFMemoryFriendly(disassemblyFiles, "dyndis"):
        #    prettyPrint("TF-IDF features were successfully generated")
        #else:
        #    prettyPrint("Could not generated TF-IDF features", "warning")
        #    return False
         
    except Exception as e:
        prettyPrint("Error encountered in \"extractTFIDF\": %s" %e, "error")
        cleanUp()
        return False

    return True

def extractResourceUtil(sourceFiles):
    """ Extracts resource utilization features from all files in a given directory """
    try:
        for targetFile in sourceFiles:
            outFile = compileFile(targetFile)
            if outFile == "":
                prettyPrint("Unable to compile \"%s\". Skipping" % targetFile, "warning")
                continue

            outFile = compileFile(targetFile)
            timeArgs = ["/usr/bin/time", "-f", "[%D,%E,%F,%I,%K,%M,%O,%P,%R,%S,%U,%W,%X,%Z,%c,%e,%k,%p,%r,%s,%t,%w]", outFile]
            
            prettyPrint("Running \"%s\" and monitoring its resource usage" % targetFile, "debug")
            timeOutput = subprocess.Popen(timeArgs, stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
            sideEffectsVector = timeOutput.split('\n')[-2]
            # Type 3 - dynamic side effects
            prettyPrint("Dumping resource utilization features to \"%s\"" % targetFile.replace(".c", ".util"))
            sideEffectFile = open(targetFile.replace(".c", ".util"), "w")
            sideEffectFile.write(sideEffectsVector)
            sideEffectFile.close()
            
            os.unlink(outFile)

    except Exception as e:
        prettyPrint("Error encountered: %s" % e, "error")
        return False
        
    return True
    
def extractTraces(sourceFiles):
    """ Extracts traces from all files in a given directory and saves them as parameterized and unparametrized alpha sequences """
    try:
        for targetFile in sourceFiles:
            # Make sure the GDB script is there
            if not os.path.exists("script"):
                prettyPrint("The GDB script file was not found. Creating one", "warning")
                generateGDBScript()

            outFile = compileFile(targetFile)
            if outFile == "":
                prettyPrint("Unable to compile \"%s\". Skipping" % targetFile, "warning")
                continue        
    
            prettyPrint("Launching the GDB script. Release the Kraken!!")
            print subprocess.Popen(["gdb", "--batch-silent", "-x", "script", outFile], stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
            # Get the instruction trace of the process from "gdb.txt" and extract features from it
            if os.path.exists("gdb.txt"):
                # Store the contents of "gdb.txt" as disassembly for further processing
                prettyPrint("Dumping dynamic disassembly to \"%s\"" % targetFile.replace(".c", ".dyndis"), "debug")
                gdbFile = open(targetFile.replace(".c", ".dyndis"), "w")
                gdbFile.write(open("gdb.txt").read())
                gdbFile.close()
                instructionTrace = loadInstructionTrace()            

            instructionTraceString = itraceToStr(instructionTrace) # TODO: A string-format of the instruction trace for word frequency calculation
            
            prettyPrint("Converting the instruction trace to an alpha sequence", "debug")
            instructionAlphaSequence = sequenceToAlpha( instructionTraceString ) # Convert to alpha sequence
            # Store the instruction trace's alpha sequence to file
            prettyPrint("Saving the alpha sequence to \"%s\"" % targetFile.replace(".c", ".seq"))
            open("%s" % targetFile.replace(".c",".seq"), "w").write(instructionAlphaSequence)
            prettyPrint("Successfully written the alpha sequence to \"%s\"" % targetFile.replace(".c", ".seq"), "info2")
                
            prettyPrint("Converting the instruction trace to an alpha sequence with params", "debug")
            instructionAlphaSequenceParams = sequenceToAlphaParams( instructionTrace ) # Alpha sequence with operands
            # Store the parametrized sequence to file
            prettyPrint("Saving the parametrized syscall sequence to \"%s\"" % targetFile.replace(".c", ".parseq"))
            open("%s" % targetFile.replace(".c", ".parseq"), "w").write(instructionAlphaSequenceParams)
            prettyPrint("Successfully written the parametrized sequence to \"%s\"" % targetFile.replace(".c",".parseq"), "info2")

            cleanUp()
    
    except Exception as e:
        prettyPrint("Error encountered: %s" % e, "error")
        return False
        
    return True
    
def extractInstrSwitchFrequency(sourceFiles):
    """ Extracts instruction-switching frequency features from all files in the given directory """
    try:
        for targetFile in sourceFiles:
            # Make sure the GDB script is there
            if not os.path.exists("script"):
                prettyPrint("The GDB script file was not found. Creating one", "warning")
                generateGDBScript()

            outFile = compileFile(targetFile)
            if outFile == "":
                prettyPrint("Unable to compile \"%s\". Skipping" % targetFile, "warning")
                continue

            prettyPrint("Launching the GDB script. Release the Kraken!!")
            subprocess.call(["gdb", "--batch-silent", "-x", "script", outFile])
            # Get the instruction trace of the process from "gdb.txt" and extract features from it
            if os.path.exists("gdb.txt"):
                instructionTrace = loadInstructionTrace()            
            instructionTraceString = itraceToStr(instructionTrace) # TODO: A string-format of the instruction trace for word frequency calculation
            
            prettyPrint("Extractig numerical features from the instruction trace", "debug")
            instructionTraceFeatures = extractFeaturesFromITrace(instructionTrace) # TODO: A list of numerical features extracted from the instruction trace
            # Save instruction switches to file
            fileName = "%s" % targetFile.replace(".c", ".freq")
            dataFileHandle = open(fileName, "w")
            prettyPrint("Dumping instruction-switch frequency features to \"%s\"" % targetFile.replace(".c", ".freq"))
            dataFileHandle.write(str(instructionTraceFeatures))
            dataFileHandle.close()

            cleanUp()    
    
    except Exception as e:
        prettyPrint("Error encountered: %s" % e, "error")
        return False
        
    return True
    
def extractHMMFeatures(sourceFiles):
    """ Extracts HMM-similarity features from all files in a given directory """
    allTraces = [] # List to store all traces for the HMM-similarity extraction     
    try:
        for targetFile in sourceFiles:
            if os.path.exists(targetFile.replace(".c", ".seq")):
                instructionAlphaSequence = open(targetFile.replace(".c", ".seq")).read()
                allTraces.append( (instructionAlphaSequence, targetFile.replace(".c", ".hmm"), loadLabelFromFile(targetFile.replace(".c", ".metadata"))[0])) #TODO: Append a tuple of (trace, filename, cluster) for each data sample
        if len(allTraces) < 1:
            prettyPrint("No traces to process for HMM-similarity feature extraction. Skipping", "warning")
        else:
            allClusters = []
            # Retrieve list of clusters
            prettyPrint("Retrieving clusters")
            for trace in allTraces:
                if not trace[2] in allClusters:
                    allClusters.append(trace[2])
            # Gather traces belonging to different clusters
            clusterTraces = []
            for cluster in allClusters:
                currentCluster = []
                for trace in allTraces:
                    if trace[2] == cluster:
                        currentCluster.append(trace[0])
                clusterTraces.append(currentCluster)
                prettyPrint("Retrieved %s instances for cluster %s" % (len(currentCluster), cluster))
            # Should wind up with list of lists each of which depict traces of a cluster
            allHMMs = []
            for cluster in allClusters:
                # Build HMM for each cluster and use it to calculate likelihoods for all instances
                prettyPrint("Building HMM for cluster \"%s\"" % cluster)
                trainingSequences =  clusterTraces[ allClusters.index(cluster) ]
                # Retrieve number of observations
                observations = []
                for sequence in trainingSequences:
                    for o in sequence:
                        if o not in observations:
                            observations.append(o)
                # Prepare matrices for HMM
                A = numpy.random.random((len(allClusters), len(allClusters))).tolist()
                B = numpy.random.random((len(allClusters), len(observations))).tolist()
                Pi = numpy.random.random((len(allClusters),)).tolist()
                sigma = ghmm.Alphabet(observations)
                # Build HMM and train it using Baum-Welch algorithm
                clusterHMM = ghmm.HMMFromMatrices(sigma, ghmm.DiscreteDistribution(sigma), A, B, Pi)
                clusterHMM.baumWelch(ghmm.SequenceSet(clusterHMM.emissionDomain, trainingSequences))
                # Add that to list of all HMM's
                allHMMs.append((clusterHMM, observations))
            # Finally, for every trace, calculate the feature vectors
            prettyPrint("Calculating similarity features for traces")
            for trace in allTraces:
                featureVector = []
                for hmm in allHMMs:
                    # Make sure sequences contains observations supported by the current HMM
                    sequence = []
                    for obs in trace[0]:
                        if obs in hmm[1]:
                            sequence.append(obs)
                    # Calculate the likelihood
                    sequence = ghmm.EmissionSequence(ghmm.Alphabet(hmm[1]), sequence)
                    featureVector.append(hmm[0].loglikelihood(sequence))
                    featureFile = open(trace[1], "w")
                    featureFile.write(str(featureVector))
                    featureFile.close()
        #############################################################################

    except Exception as e:
        prettyPrint("Error encoutered: %s" % e, "error")
        return False
        
    return True 
   
def extractDifference(sourceDir, datatype):
    """ Extracts the differences between original programs and their obfuscated versions """
    try:
        # Retrieve original programs
        prettyPrint("Loading the list of original programs")
        originalFiles = list(set(glob.glob("%s/*.%s" % (sourceDir, datatype))) - set(glob.glob("%s/*_*.%s" % (sourceDir, datatype))))
        prettyPrint("Successfully retrieved %s original programs" % len(originalFiles))
        counter = 0
        allDisassemblies = [] # To hold the difference disassembly files for TF-IDF extraction
        for originalFile in originalFiles:
            # Retrieve obfuscated versions of each original file
            obfuscatedVersions = glob.glob("%s_*.%s" % (originalFile.replace(".%s" % datatype, ""), datatype))
            prettyPrint("Successfully retrieved %s obfuscated versions for \"%s\"" % (len(obfuscatedVersions), originalFile), "debug")
            originalSet = set(open(originalFile).read().split('\n')) # Set of instructions in original file
            for obfuscated in obfuscatedVersions:
                obfuscatedSet = set(open(obfuscated).read().split('\n')) # Set of instructions in obfuscated version
                diffSet = set.difference(obfuscatedSet, originalSet) # Tj = Pi' - Pi
                # Save difference instructions (order doesn't matter as it cannot be run anyway)
                diffFile = open(obfuscated.replace(datatype, "%sdiff" % datatype), "w")
                for instruction in list(diffSet):
                    diffFile.write("%s\n" % instruction)
                diffFile.close()
                if os.path.exists(obfuscated.replace(datatype, "%sdiff" % datatype)) and os.path.getsize(obfuscated.replace(datatype, "%sdiff" % datatype)) > 0:
                    # Make sure it exists and not empty
                    counter += 1
                    
        prettyPrint("Successfully generated %s difference files" % counter)
              
        sourceFiles = glob.glob("%s/*_*.%sdiff" % (sourceDir, datatype))
        for targetFile in sourceFiles:
            allDisassemblies.append(open(targetFile).read())
                                
        # Now perform TF-IDF on them
        vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, max_features=1000, stop_words=[",","%","(",")",",",":","\n","$"], norm='l2', smooth_idf=True, use_idf=True, sublinear_tf=False)
        X = vectorizer.fit_transform(allDisassemblies)
        for targetFile in sourceFiles:
            # Get the feature vector
            featureVector = X.toarray()[ sourceFiles.index(targetFile),:].tolist()
            # Save it to file 
            featureFile = open(targetFile.replace("%sdiff" % datatype, "%sdiffidf" % datatype), "w")
            featureFile.write(str(featureVector))
            featureFile.close()
        
    except Exception as e:
        prettyPrint("Error encountered: %s" % e, "error")
        return False

    return True
    
def extractDifferenceFromTraces(sourceDir, datatype):
    """ Extracts the differences between original programs and their obfuscated versions represented as encoded instruction traces """
    try:
        # Retrieve original programs
        prettyPrint("Loading the list of original programs")
        originalFiles = list(set(glob.glob("%s/*.%s" % (sourceDir, datatype))) - set(glob.glob("%s/*_*.%s" % (sourceDir, datatype))))
        prettyPrint("Successfully retrieved %s original programs" % len(originalFiles))
        counter = 0
        allTraces = [] # To hold the difference sequences for TF-IDF extraction
        for originalFile in originalFiles:
            # Retrieve obfuscated versions of each original file
            obfuscatedVersions = glob.glob("%s_*.%s" % (originalFile.replace(".%s" % datatype, ""), datatype))
            prettyPrint("Successfully retrieved %s obfuscated versions for \"%s\"" % (len(obfuscatedVersions), originalFile), "debug")
            originalTrace = list(open(originalFile).read())
            for obfuscated in obfuscatedVersions:
                obfuscatedTrace = list(open(obfuscated).read()) 
                # Calculate the difference between two sequences
                indexMax = min(len(originalTrace), len(obfuscatedTrace))
                diffTrace = [] + obfuscatedTrace
                for index in range(indexMax):
                    if originalTrace[index] == diffTrace[index]:
                        diffTrace[index] = "_"
                diffFile = open(obfuscated.replace(datatype, "%sdiff" % datatype), "w")
                for instruction in list(diffTrace):
                    diffFile.write("%s\n" % instruction)
                diffFile.close()
                if os.path.exists(obfuscated.replace(datatype, "%sdiff" % datatype)) and os.path.getsize(obfuscated.replace(datatype, "%sdiff" % datatype)) > 0:
                    # Make sure it exists and not empty
                    counter += 1
                    
        prettyPrint("Successfully generated %s difference files" % counter)
              
        sourceFiles = glob.glob("%s/*_*.%sdiff" % (sourceDir, datatype))
        for targetFile in sourceFiles:
            allTraces.append(open(targetFile).read())
                                
        # Now perform TF-IDF on them
        vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, max_features=1000, stop_words=[",","%","(",")",",",":","\n","$"], norm='l2', smooth_idf=True, use_idf=True, sublinear_tf=False)
        X = vectorizer.fit_transform(allTraces)
        for targetFile in sourceFiles:
            # Get the feature vector
            featureVector = X.toarray()[ sourceFiles.index(targetFile),:].tolist()
            # Save it to file 
            featureFile = open(targetFile.replace("%sdiff" % datatype, "%sdiffidf" % datatype), "w")
            featureFile.write(str(featureVector))
            featureFile.close()
        
    except Exception as e:
        prettyPrint("Error encountered: %s" % e, "error")
        return False

    return True
   
def extractTritonFeatures(sourceDir):
    """ Extracts some dynamic features from files in a specific directory using Triton """
    try:
        # Retrieve source files
        sourceFiles = glob.glob("%s/*.c" % sourceDir)
        if len(sourceFiles) < 1:
            prettyPrint("No source files were found under \"%s\"" % sourceDir, "warning")
            return False
        # Iterate on all files
        for targetFile in sourceFiles:
            # Check if there is a Triton file already
            if os.path.exists(targetFile.replace(".c", ".triton")):
                prettyPrint("A \"Triton\" file already exists for \"%s\". Skipping" % targetFile, "warning")
                continue
            # Compile the source files first
            outFile = compileFile(targetFile)
            # Run it using Triton and its python script
            tritonCmd = ["sudo", "./triton", "triton_script.py", "./%s" % outFile]
            #print str(tritonCmd)[1:-1].replace(",","")
            prettyPrint("Launching \"Triton\" with command %s" % (tritonCmd), "debug")
            tritonFeatures = subprocess.Popen(tritonCmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
            if tritonFeatures.find("Output:") == -1:
                prettyPrint("Unable to parse the output from \"Triton\": %s. Skipping" % tritonFeatures, "warning")
            else:
                tritonFeatures = tritonFeatures[tritonFeatures.find("Output:")+len("Output:"):]
                # Save the list into a file
                tritonFile = open("%s.triton" % targetFile.replace(".c", ""), "w")
                tritonFile.write(tritonFeatures)
                tritonFile.close()
            # Clean up
            cleanUp()

        prettyPrint("Successfully generated %s \"Triton\" features files" % len(glob.glob("%s/*.triton" % sourceDir)))

    except Exception as e:
        prettyPrint("Error encountered in \"extractTritonFeatures\": %s" % e, "error")
        return False

    return True

class MyCorpus(object):
    """ Helper class for the gensim-based TF-IDF extraction """
    def __init__(self, docs):
        self.documents = docs
        self.tokens = dictionary.Dictionary()
        # Retrieve tokens form documents, populating the tokens dictionary
        for doc in self.documents:
            content = [[word for word in open(doc).read().lower().split() if word not in [",","%","(",")",",",":","\n","$"]]]
            self.tokens.add_documents(content)
        print "[*] Retrieved %s tokens from %s documents in the corpus" % (len(self.tokens), len(self.documents))

    def  __iter__(self):
        # Iterate over documents in the corpus retrurning their token counts
        for doc in self.documents:
            yield self.tokens.doc2bow(open(doc).read().lower().split())#, return_missing=True)

def cmpTuple(x,y):
#    if type(x) != tuple or type(y) != tuple or not len(x) == len(y) == 2:
#        return 0
    if x[1] > y[1]:
        return -1
    elif x[1] < y[1]:
        return 1
    else:
        return 0 

def getTupleKey(l, k):
    if len(l) < 1:
        return 0
    for element in l:
        if type(element) == tuple and len(element) == 2:
            if element[0] == k:
                return element[1]
    return 0

def extractTFIDFMemoryFriendly(source, fileextension, maxfeatures=128, outextension="tfidf"):
    """ Extracts TF-IDF features from corpus using the memory friendly gensim library """
    try:
        # Retrieve files (source can be a path or a list of files names)
        if type(source) == str:
            # Case 1: a path to a directory
            allfiles = glob.glob("%s/*.%s" % (source, fileextension))
        elif type(source) == list:
            # Case 2: a list of file names e.g. in 36-4 experiment
            allfiles = source
        else:
            prettyPrint("Unable to process sources of type \"%s\"" % str(type(source)), "warning")
            return False
        # Sort the list either way
        allfiles.sort() # Can be removed 
        if len(allfiles) < 1:
            prettyPrint("No files of extension \"%s\" could be found in \"%s\"" % (fileextension, source), "warning")
            return False

        prettyPrint("Successfully retrieved %s files" % len(allfiles))
        # Now instantiate an instance of the MyCorpus class
        corpus_mem_friendly = MyCorpus(allfiles)
        # Save the tokens to file and load them again just to get the cross-document count (:s)
        filename = "corpus_%s_%s" % (str(int(time.time())), fileextension)
        corpus_mem_friendly.tokens.save_as_text(filename)
        tokens = open(filename).read().split('\n')
        tokenTuples = []
        for t in tokens:
            if len(t) > 1:
                tokenTuples.append((int(t.split('\t')[0]), int(t.split('\t')[2])))
        # Now sort them descendingly
        prettyPrint("Sorting the tokens according to their document frequency")
        #print tokenTuples #TODO: Remove me!!
        tokenTuples.sort(cmp=cmpTuple)

        # Build a list of vectors
        allVectors = [v for v in corpus_mem_friendly]    

        # Build a numpy matrix of zeros
        X = numpy.zeros((len(allfiles), maxfeatures))

        # Go over the first [maxfeatures] of the tokenTuples and populate the matrix
        prettyPrint("Picking the best %s features from the sorted tokens list" % maxfeatures)
        for vectorIndex in range(len(allVectors)):
            prettyPrint("Processing vector #%s out of %s vectors" % (vectorIndex+1, len(allVectors)))
            for featureIndex in range(maxfeatures):
                # a. Get the token key
                tokenKey = tokenTuples[featureIndex][0]
                #print allVectors[vectorIndex], tokenKey, getTupleKey(allVectors[vectorIndex], tokenKey)
                X[vectorIndex][featureIndex] = getTupleKey(allVectors[vectorIndex], tokenKey)

        #print corpus_mem_friendly.tokens.token2id
        #print tokenTuples
        #print X
           
        # Now apply the TF-IDF transformation
        optimusPrime = TfidfTransformer()
        prettyPrint("Extracting TF-IDF features")
        X_tfidf = optimusPrime.fit_transform(X)

        prettyPrint("Saving TF-IDF vectors to \"%s\" files" % outextension)
        for doc_index in range(len(allfiles)):
            tfidf_file = open(allfiles[doc_index].replace(fileextension, outextension), "w")
            tfidf_file.write(str(X_tfidf.toarray()[doc_index,:].tolist()))
            tfidf_file.close()

        os.unlink(filename) 

    except Exception as e:
        prettyPrint("Error encountered in \"extractTFIDFMemoryFriendly\": %s" % e, "error")
        return False

    return True
