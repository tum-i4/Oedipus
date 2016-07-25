#!/usr/bin/python

import subprocess, threading, time, os
from itertools import product
from Oedipus.utils.data import *
from Oedipus.utils.graphics import *

# The new static version of Tigress transformations
tigressTransformations = {
"Virtualize": {"VirtualizeDispatch": ["switch", "direct", "indirect", "call", "ifnest", "linear", "binary", "interpolation"], "VirtualizeOperands":[ "stack", "registers"], "VirtualizePerformance":["IndexedStack", "PointerStack", "AddressSizeShort", "AddressSizeInt", "AddressSizeLong", "CacheTop"]},
"Jit":{"JitEncoding":["hard"], "JitCopyKinds":["counter", "counter_signal", "bitcopy_unrolled", "bitcopy_loop", "bitcopy_signal"]},
"AddOpaque": {"AddOpaqueKinds": ["call", "bug", "true", "junk", "fake"], "AddOpaqueCount": ["1","2","3","4","5"]},
"EncodeLiterals":{"EncodeLiteralsKinds": ["integer", "string"]},
"EncodeArithmetic": {"EncodeArithmeticKinds":["integer"]},
"Flatten":{"FlattenDispatch":["switch","goto","indirect", "call"], "FlattenOpaqueStructs":["list","array"]}
}
numPrograms = 0
currentThreads = 0

def _permutations(r=None):
    """ Returns an iterator of permutations of an iterable's members """
    pool = tuple(tigressTransformations.keys()) # Get it from the global variable
    n = len(pool)
    r = n if r is None else r
    for indices in product(range(n), repeat=r):
        if len(set(indices)) == r:
            yield tuple(pool[i] for i in indices)

def generateMultipleObfuscations(currentFile, tigressDir, obfuscationLevel=1, obfuscationFunction="main"):
    """ Generates obfuscation versions of a program with varying obfuscation levels """
    try:
        global numPrograms
        # Get the permutations first
        obfuscations = _permutations(obfuscationLevel)
        if obfuscationLevel > 1:
            obfuscations += tuple(("Virtualize;"*obfuscationLevel).split(';')[:-1])
            obfuscations += tuple(("Flatten;"*obfuscationLevel).split(';')[:-1])
            obfuscations += tuple(("EncodeArithmetic;"*obfuscationLevel).split(';')[:-1])

        for permutation in obfuscations:
            # Build the obfuscation command
            tigressCmd = ["tigress"]
            # Generate the timesamp for the file to generate
            currentFileTimestamp = str(time.time()).replace(".", "_")
            jitFlag = False
            for transformation in permutation:
                if transformation == "AddOpaque":# or transformation == "EncodeLiterals":
                    tigressCmd += ["--Transform=InitOpaque", "--Functions=%s" % obfuscationFunction]
                # Add the necessary "#include" for Jitting to a copy of the file
                if transformation == "Jit":
                    fileContent = open(currentFile).read()
                    fileContent = "#include \"%s/jitter-amd64.c\"\n" % tigressDir + fileContent
                    copyFile = open(currentFile.replace(".c", "_jitcopy.c"), "w")
                    copyFile.write(fileContent)
                    copyFile.close()
                    jitFlag = True

                # Add the transformation, its paramters, and the target function(s)
                tigressCmd.append("--Transform=%s" % transformation)
                tigressCmd.append("--Functions=%s" % obfuscationFunction)
            
            #TODO: prepends "obf" to the generated file name
            tigressCmd.append("--FilePrefix=obf%s" % obfuscationLevel+1) 

            # Specify the output file
            if jitFlag:
                tigressCmd.append("--out=%s_%s.c" % (currentFile.replace(".c", ""), currentFileTimestamp))
                tigressCmd.append(currentFile.replace(".c", "_jitcopy.c")) # Add the target file 
            else:
                tigressCmd.append("--out=%s_%s.c" % (currentFile.replace(".c", ""), currentFileTimestamp))
                tigressCmd.append(currentFile) # Add the target file

            
            # Execute the command
            prettyPrint("Generating an obfuscated version of \"%s\" using the command \"%s\"" % (currentFile, tigressCmd), "debug")
            tigressOutput = subprocess.Popen(tigressCmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
            # Check if the obfuscated file got generated
            if not os.path.exists("%s_%s.c" % (currentFile.replace(".c",""), currentFileTimestamp)) or os.path.getsize("%s_%s.c" % (currentFile.replace(".c",""), currentFileTimestamp)) == 0:
                print(tigressOutput)
            else:
                # Save/update the label file
                label = "_".join(permutation)
                if os.path.exists(currentFile.replace(".c", ".label")):
                    # Add the newly-generated transformations to the existing label
                    currentFileLabelContent = open(currentFile.replace(".c", ".label")).read().replace("\n","")
                    currentFileLabelContent += "_%s" % label
                    open("%s_%s.label" % (currentFile.replace(".c", ""), currentFileTimestamp), "w").write(currentFileLabelContent)
                else:
                    currentFileLabel = open("%s_%s.label" % (currentFile.replace(".c",""), currentFileTimestamp), "w")
                    currentFileMetadata.write(str(tigressCmd))
                    currentFileMetadata.close()

                # Increment the number of obfuscated programs
                numPrograms += 1
            # Delete the jitcopy file, if exists
            if os.path.exists(currentFile.replace(".c", "_jitcopy.c")):
                os.unlink(currentFile.replace(".c", "_jitcopy.c"))

    except Exception as e:
        prettyPrint("Error encountered in \"generateMultipleObfuscations\": %s" % e, "error")
        return False

    return True

def generateObfuscatedPrograms(sourceFiles, tigressDir="/opt/tigress-unstable", obfuscationLevel=1, obfuscationFunction="main"):
    """ Generates obfuscated programs using supported "Tigress" transformations """
    #numPrograms = len(sourceFiles)
    for currentFile in sourceFiles:
        # Write a metadata file for the plain file
        plainLabelFile = open(currentFile.replace(".c", ".label"), "w")
        plainLabelFile.write("Ident")
        plainLabelFile.close()
        # Multi-level, single-threaded obfuscation 
        generateMultipleObfuscations(currentFile, tigressDir, obfuscationLevel, obfuscationFunction)

    return numPrograms
