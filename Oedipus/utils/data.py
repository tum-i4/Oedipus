#!/usr/bin/python

from Oedipus.utils.graphics import prettyPrint
from Levenshtein import distance
import glob, os, sys, re
import numpy

###########################################
# Define the global variables and classes #
###########################################

# Used to transform instruction traces to alphabetic (protein-like) sequences
availableLetters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
availableOperandLetters = ["!", "#", "$", "^", "&", "*", "(", ")", "-", "+", "=", "~", "?", ":", ";", "{", "}", "[", "]", "|", "<", ">", ",", "."]
sequenceAlphaMap = {}
sequenceAlphaOperandMap = {}
encodedOpCodes = []
encodedOperands = ["reg", "imm", "mem", "lbl"]

""" To be removed because it is not actually used """
class DataSample():
    def __init__(self, sampleName, sampleFeatures, sampleCluster, sampleParams={}):
        self.sampleName = sampleName
        self.sampleFeatures = sampleFeatures
        self.sampleCluster = sampleCluster
        self.sampleParams = sampleParams
""" To be removed because it is not actually used """

#################################
# Define the associated methods #
#################################
def loadArgumentsFromKLEE(fileName):
    """ Parses KLEE testcase, saves arguments to file, and returns a list of retrieved arguments """
    fileContent = open(fileName, "rb").read().split("\n")
    if len(fileContent) < 1:
        prettyPrint("KLEE testcase file is empty", "warning")
        return [], ""
    args, numArgs = [], 0
    argFlag = False
    # Retrieve the number of arguments
    for row in fileContent:
        if row.find("arg") != -1 and row.find("n_args") == -1 and row.find("--sym-args") == -1:
            argFlag = True
        elif row.find("data: ") != -1 and argFlag:
            argIndex = row.find("data: ") + len("data: ")
            args.append(row[argIndex:].replace("'","").decode("string_escape"))
            argFlag = False
    # Now write arguments to file
    inputFile = open(fileName.replace(".txt",".input"), "wb")
    for arg in args:
        inputFile.write(arg)
        inputFile.write(" ")
    inputFile.close()

    return args, fileName.replace(".txt", ".input")

def loadFeaturesFromDir(dirName, dataType, dataLabel="metadata"):
    """ Loads features all files in a directory into two lists """
    # Retrieve all files
    dataFiles = glob.glob("%s/*.%s" % (dirName, dataType))
    return loadFeaturesFromList(dataFiles, dataType, dataLabel)
        
def loadFeaturesFromFile(fileName):
    """ Loads features from a file into a list """
    features = []
    rawData = open(fileName).read().split('\n')
    for f in rawData:
        #if f.replace(".","").isdigit():
        if f != '':
            features.append(float(f))
    return features

def loadFeaturesFromList(dataFiles, dataType, labelExtension="metadata", classReference=[]):
    """ Loads features from a list of files """
    features = []
    # Retrieve all files
    if len(dataFiles) < 1:
        prettyPrint("No data files of type \"%s\" were found." % dataType, "warning")
        return numpy.array([]), numpy.array([])
    # Iterate over files adding their values to an array
    dataPoints, dataLabels, allClasses = [], [], []
    labelFile = "" # TODO: Again for KLEE test files
    for dataFile in dataFiles:
        currentExtension = dataFile[dataFile.rfind("."):]
        if labelExtension == "label":
            # TODO: Accomodate for the KLEE files
            if dataFile.find("test") != -1:
                labelFile = dataFile[:dataFile.rfind("_test")] + ".label"
                if not os.path.exists(labelFile):
                    prettyPrint("Could not find a label file for \"%s\". Skipping" % dataFile, "warning")
                    continue
            else:
                if not os.path.exists(dataFile.replace(dataType, "label")):
                    prettyPrint("Could not find a label file for \"%s\". Skipping" % dataFile, "warning")
                    continue
 
        dataFile = dataFile.replace(currentExtension,".%s" % dataType) # Make sure we're loading from the right extension

        if dataType.find("tfidf") != -1 or dataType == "freq" or dataType == "util" or dataType == "hmm":
            # Load features as numerical
            dataPoints.append([float(x) for x in open(dataFile).read()[1:-1].split(',')])
            #print dataPoints
        elif dataType == "triton":
           # Load features as numerical/nominal
           content = open(dataFile).read().replace("\n", "").replace(" ", "")[1:-1]
           features = content.split(",")
           for index in range(len(features)):
               features[index] = features[index].replace("'","")
               if features[index].isdigit():
                   features[index] = int(features[index])
               elif features[index].find(".") != -1:
                   features[index] = float(features[index])
               else:
                   # Numerizing "Yes" and "No"
                   if features[index].lower() == "yes":
                       features[index] = 1.0
                   else:
                       features[index] = 0.0
           # Append to dataPoints
           dataPoints.append(features)
        elif dataType == "seq" or dataType == "parseq":
            # Load features as sequence of strings
            dataPoints.append(open(dataFile).read())
        # Also add the class label
        if labelExtension == "label":
            if labelFile != "":
                currentClass, currentParams = loadLabelFromFile(labelFile)
            else:
                currentClass, currentParams = loadLabelFromFile(dataFile.replace(".%s" % dataType, ".label"))
        elif labelExtension == "metadata":
            currentClass, currentParams = loadLabelFromFile(dataFile[:dataFile.rfind("_test")] + ".metadata")
            for attribute in currentParams:
                currentClass += "_%s_%s" % (attribute, currentParams[attribute])
        currentClass = currentClass.replace(" ","") # Get rid of any spaces
        # Translate that to integers
        if currentClass in classReference:
            dataLabels.append(classReference.index(currentClass))
        else:
            classReference.append(currentClass)
            dataLabels.append(classReference.index(currentClass)) # Add an index as the class label
    # Now return the data points and labels as lists
    return dataPoints, dataLabels, classReference

def loadLabelFromFile(fileName):
    """ Loads clusters from file into a string """
    cluster, params = "", {}
    if not os.path.exists(fileName):
        prettyPrint("Could not find \"%s\". Skipping" % fileName, "warning")
        return cluster, params

    rawData = open(fileName).read()
    if fileName.find(".label") != -1:
        # It's a label file
        cluster, params = rawData.split('\n')[0], {}
    else:
        # It's a metadata file
        if rawData.find("Ident") != -1:
            cluster, params = "Ident", {}
        else:
            for token in rawData[1:-1].split(","):
                if token.find("Transform") != -1:
                    cluster = cluster + token.split("=")[1].replace("'","") + "_"
                else:
                    if token.find("=") != -1 and token.find("Functions") == -1 and token.find("out") == -1:
                        key = token.split('=')[0].replace('-','').replace("'","")
                        value = token.split('=')[1].replace("'","")
                        params[key] = value
                
        cluster = cluster[:-1] if cluster[-1] == "_" else cluster # Clip any trailing underscores

    return cluster, params

def loadInstructionTrace(fileName="gdb.txt"):
    """ Parses the 'gdb.txt' file and returns assembly instructions in a list """
    iTrace = []
    # Get text from file
    allText = open(fileName).read()
    allText = allText[allText.find("Breakpoint 1"):]
    # Split into lines
    allLines = allText.split('\n')
    # Remove C/C++ lines
    for line in allLines:
        if line.find("=>") != -1:
            iTrace.append((line.split('\t')[-1].split(' ')[0], line.split('\t')[-1].split(' ')[-1].split(',')))
    return iTrace

def loadAlphaSequences(fileName, sequenceSize=0):
    """ Loads alpha sequences from a file into a list of characters """
    alphaSequence = []
    if not os.path.exists(fileName):
        prettyPrint("File \"%s\" was not found" % fileName, "warning")
    rawSequence = open(fileName).read()
    for alpha in rawSequence:
        if alpha != '' and alpha != '\n':
            alphaSequence.append(alpha)
    if sequenceSize == 0 or sequenceSize > len(alphaSequence):
        return alphaSequence
    else:
        return alphaSequence[:sequenceSize]

def itraceToList( trace ):
    """ Converts an instruction trace into list """
    traceList = []
    for i in trace:
        traceList.append(i[0])
    return traceList

def itraceToStr( trace ):
    """ Concerts an instruction trace into a string """
    traceStr = ""
    for i in trace:
        traceStr += "%s," % i[0]
    return traceStr[:-1] # Ignore the tailing comma

def sequenceToAlpha( behavior ):
    """ Converts an instruction trace into an alphabet sequence """
    alphaSequence = ""
    global availableLetters
    global sequenceAlphaMap

    try:
        if type(behavior) == str:
            behavior = behavior.split(',')

        for action in behavior:
            if not action in sequenceAlphaMap.keys():
                sequenceAlphaMap[ action ] = availableLetters.pop(0)
            alphaSequence += sequenceAlphaMap[ action ]

    except Exception as e:
        prettyPrint("Error encountered while converting trace into alpha sequence: %s" % e, "error")
        prettyPrint("Length of current sequence is \"%s\"" % len(alphaSequence))

    return alphaSequence

def getOperandType( operand ):
    """ Return the type of passed operand i.e. reg, mem, imm, etc. """
    if operand.find("$") != -1:
        return "imm"
    elif operand.find("(") != -1 or operand.find("0x") != -1:
        return "mem"
    elif operand.find("%") != -1:
        return "reg"
    else:
        return "lbl"

def sequenceToAlphaParams( trace ):
    """ Converts an instruction trace into a parameterized alphabet sequence """
    global availableLetters
    global availableOperandLetters
    alphaSequence = ""
    key, value = "", ""
    for operation in trace:
        # Step 1 - Retrieve the key and transform it into an alpha letter
        key = operation[0]
        # (1.1) Check whether it exists in the sequence-alpha map
        if not key in sequenceAlphaMap.keys():
            sequenceAlphaMap[ key ] = availableLetters.pop(0)
        # Step 2 - Do the same for the operands
        operands = operation[1]
        # (2.1) Model the operands as comma-separated strings
        for op in range(len(operands)):
            value += "%s," % getOperandType( operands[op] )
        value = value[:-1]
        # (2.2) Now check whether that string exists in the sequence-alpha map of operands
        if not value in sequenceAlphaOperandMap.keys():
           sequenceAlphaOperandMap[ value ] = availableOperandLetters.pop(0)

       # Step 3- Add the key and value to the trace
        alphaSequence+= "%s%s" % (sequenceAlphaMap[ key ], sequenceAlphaOperandMap[ value ])
        # Reset the values of key and value
        key, value = "", ""
    return alphaSequence[:-1]

def encodeSequence(rawSequence, sequenceType="disassembly"):
    """ Encodes a sequence """
    encodedSequence = []
    global encodedOpCodes, encodedOperands
    if sequenceType == "disassembly":
        for instruction in rawSequence:
            # This looks a bit different. It's a string rather than a (opcode, [operands]) tuple
            encodedInstruction = ""
            # Split on spaces
            insList = instruction.split(" ")
            # Encode the opcode
            if not insList[0] in encodedOpCodes:
                encodedOpCodes.append(insList[0]) # Guaranteed to be the opcode
            encodedInstruction += str(encodedOpCodes.index(insList[0]))
            # Now for the operands
            for item in insList[1:]:
                if item != "":
                    operands = item.split(",")
                    for op in operands:
                        encodedInstruction += str(encodedOperands.index(getOperandType(op)))
            encodedSequence.append(encodedInstruction)
    else:
        for instruction in rawSequence:
            encodedInstruction = ""
            # Encode the opcode
            if not instruction[0] in encodedOpCodes:
                encodedOpCodes.append(instruction[0])
            encodedInstruction += str(encodedOpCodes.index(instruction[0]))
            # Now encode the operands
            for op in instruction[1]:
                encodedInstruction += str(encodedOperands.index(getOperandType(op)))
            encodedSequence.append(encodedInstruction)

    return encodedSequence

def parseDisassemblyFile(fileName):
    """ Parses a disassembly file and returns instructions in a list """
    instructions = []
    lines = open(fileName).read().split('\n')
    # Parse the lines
    stIndex, endIndex = 0, 0
    for i in range(len(lines)):
        if lines[i].find("<") != -1:
            stIndex = lines[i].index(">")+2
            if lines[i][stIndex:].find("#") != -1:
                endIndex = lines[i][stIndex:].rindex("#") - 1
            elif lines[i][stIndex:].find("<") != -1:
                endIndex = lines[i][stIndex:].rindex("<") - 1
            else:
                endIndex = len(lines[i][stIndex:])
            instructions.append(lines[i][stIndex:][:endIndex])
            stIndex, endIndex = 0, 0 # Reset indices
    return instructions

def flipSign(data, sign="+"):
    """ Flips the sign of numerical elements in the dataset to <sign> """
    tempData = [] + data # Create a new copy and keep original data safe
    for vectorIndex in range(len(tempData)):
        for featureIndex in range(len(tempData[vectorIndex])):
            if tempData[vectorIndex][featureIndex] < 0 and sign == "+":
                tempData[vectorIndex][featureIndex] *= -1
            elif tempData[vectorIndex][featureIndex] > 0 and sign == "-":
                tempData[vectorIndex][featureIndex] *= -1 

    return tempData 

def filterTraces(sourceDir, inExtension, filterMode, outExtension, targetFunction="main"):
    """ Filters the GDB generated traces according to the supplied [filterMode] """
    immReg = r'\$0x\w+'
    memReg = r'0x\w+'

    # Retrieve list of files from input dir
    allfiles = glob.glob("%s/*.%s" % (sourceDir, inExtension))
    if len(allfiles) < 1:
        prettyPrint("Unable to retrieve \"*.%s\" from \"%s\"" % (inExtension, sourceDir), "warning")
        return False

    prettyPrint("Successfully retrieved %s \"*.%s\" from \"%s\"" % (len(allfiles), inExtension, sourceDir), "debug")
    filecounter = 0
    previousline = ""
    # Loop on retrieved file and filter their content
    for inputfile in allfiles:
        prettyPrint("Processing file: %s, #%s out of %s" % (inputfile, filecounter+1, len(allfiles)), "debug")
        content = open(inputfile).read()
        outputfile = open(inputfile.replace(inExtension, outExtension), "w")
        alllines = content.split('\n')
        inMain = False
        if inExtension.find("objdump") != -1 or inExtension.find("objdumps") != -1:
            rawlines = []
            for line in alllines:
                if line.find("<%s>" % targetFunction) != -1:
                    inMain = True
                elif line.find(">:") != -1:
                    inMain = False
                if inMain and len(line.split('\t')) > 2:
                    if line.find("call") != -1 or line.find("callq") != -1:
                        functionName = line[line.rfind('<')+1:line.rfind('>')]
                        rawlines.append("%s()" % functionName)
                    else:
                        rawlines.append(line.split('\t')[-1])
        else:
            rawlines = []
            for line in alllines:
                if line.find("=>") != -1 and line.find(targetFunction) != -1:
                    rawlines.append(line[line.find(':')+1:])
                else:
                    # Not a target function
                    # Check whether it is a "call" instruction
                    if line.find("call") != -1 or line.find("callq") != -1:
                        if line.find("%") == -1:
                            functionName = line[line.rfind("<")+1:line.rfind("+")]
                            line = "%s()" % functionName

        # Now filter them
        for templine in rawlines:
            # Match and replace immediate and memory values
            # Are we allowed to filter immediate values as well?
            if filterMode.lower() == "both":
                # Yes, then get rid of the immediate first (the more specific)
                templine = re.sub(immReg, "imm", templine)
                templine = re.sub(memReg, "mem", templine)
            elif filterMode.lower() == "mem":
                # No, then check whether this is an immediate match
                if re.search(immReg, templine):
                    # ... and skip
                    pass
                else:
                    # Otherwise, just filter the memory location
                    templine = re.sub(memReg, "mem", templine)
            elif filterMode.lower() == "raw":
                # Leave both the memory and immediate values alone
                templine = templine
            else:
                prettyPrint("Unknown filter mode \"%s\". Exiting." % filterMode, "warning")

            # Remove commas
            templine = templine.replace(',', ' ')
            # Write the instruction to file
            #instruction = templine.split()
            #finalline = ""
            #for i in instruction:
            #    finalline += " %s" % i
            outputfile.write("%s\n" % templine)
    
        filecounter += 1

    prettyPrint("Successfully processed %s \"*.%s\"." % (filecounter, inExtension), "debug")
    outputfile.close()
    return True

