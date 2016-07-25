#!/usr/bin/python

import random, string, os, glob
from datetime import datetime

def checkRoot():
    if os.getuid() != 0:
        return False
    else:
        return True

def cleanUp():
    # Removes temporary files and directories generated during execution
    if os.path.exists("tempCCCC"):
        shutil.rmtree("tempCCCC")
    if os.path.exists("script"):
        os.unlink("script")
    executables = glob.glob("./*.out")
    for exe in executables:
        os.unlink(exe)
    scripts = glob.glob("./*.script")
    for script in scripts:
        os.unlink(script)
    texts = glob.glob("./*.txt")
    for text in texts:
        os.unlink(text)
    if os.path.exists("log.txt"):
        os.unlink("log.txt")
    if os.path.exists("program.txt"):
        os.unlink("program.txt")
    if os.path.exists("gdb.txt"):
        os.unlink("gdb.txt")

def getRandomNumber(length=8):
    return ''.join(random.choice(string.digits) for i in range(length))

def getRandomAlphaNumeric(length=8):
    return ''.join(random.choice(string.ascii_letters + string.digits) for i in range(length))

def getRandomString(length=8):
    return ''.join(random.choice(string.lowercase) for i in range(length))

def getTimestamp():
    return "[%s]"%str(datetime.now()).split(" ")[1]

def averageList(inputList, roundDigits=2):
   return round(float(sum(inputList))/float(len(inputList)), roundDigits)

def getOriginalFileName(fileName, fileExtension=".c"):
    """ Strips a file name off digits and underscores and returns the original file name """
    originalFileName = fileName.replace("_","").replace(fileExtension,"")
    # Now remove the digits
    for i in range(10):
       originalFileName = originalFileName.replace(str(i),"")
    # Finally, remove any previous directories
    slashIndex = originalFileName.find("/")
    while slashIndex != -1:
        originalFileName = originalFileName[slashIndex+1:]
        slashIndex = originalFileName.find("/")

    return originalFileName

def checkTestCaseSuccess(output):
    """ Searches the output of the GDB script for incentives of failure """
    incentives = ["error", "fail", "unexpected", "cannot"]
    for word in incentives:
        if output.lower().find(word) != -1:
            return False
    return True

