import matplotlib.pyplot as plt
from collections import defaultdict
import data_preparation as prepare
import visualize_helper as visualize
import signature_helper as signature
import model_helper as modelhelper
from cycler import cycler
import matplotlib as mpl
import esig.tosig as ts

windowLength = 10
mpl.rcParams['axes.prop_cycle'] = cycler(color=['r', 'g', 'r', 'g'])
mpl.rcParams['lines.linewidth'] = 2
depthSignature = 2
signatureLength = ts.sigdim(2, depthSignature)

#uhid plus the time block is the key and signature of each block of time i.e. 120 seconds is the value
#three days of data for each patient
lengthofDataToProcess = 2*60
timeBlocksCounter = int(lengthofDataToProcess/windowLength)
y_final = []
XAug = defaultdict(list)
debugCode = True
print("---------Preparing Data----------")
dataFilePath = '/Users/ravneetkaur/SignatoryProject/'
dataPreparation = prepare.DataPreparation(dataFilePath)
#below method return dictionary object with uhid_sepsis or uhid_nosepsis as key and list of data as value
X,Y,X_list = dataPreparation.prepareData('sepsis_nosepsis.csv',lengthofDataToProcess)
print('-------------Data Preparation Done------------')
#We had 25 patients prior to augmentation

#X_Aug,Y_Aug = dataPreparation.augmentData()
#We have now 50 patients post augmentation

numberOfPatients  = len(X)
dict_SignatureHR, dict_SignatureSpO2, listUHID, uhidSepsisCase, uhidNoSepsisCase,y_final,biggerListUHID,XAugHR,XAugSpO2 = signature.generateSignature(X,Y,X_list, depthSignature, windowLength,lengthofDataToProcess,dataPreparation, dataFilePath)
print('-------------Data Augmentation Done------------')
print("---------Signatures Done----------")


# visualize.generatePlotsForDataValidation(XAugHR,XAugSpO2,dict_SignatureHR,dict_SignatureSpO2,windowLength,lengthofDataToProcess,timeBlocksCounter)
# plt.show()

#
# zippedUHIDTupleDict = signature.concatenateSignatureCoefficients(dict_SignatureHR,listUHID,timeBlocksCounter)
# print("---------Signatures Coefficient Concatenation Done----------")
#
# #signature.similarityBetweenSignatureCoefficients(dict_SignatureHR,biggerListUHID,timeBlocksCounter,signatureLength)
# #print("---------Plotting similarity amongst sepsis and no-sepsis cases----------")
#
#
# signature.writeSignatureCoefficientsIntoCSV(dataFilePath, dict_SignatureHR,listUHID,timeBlocksCounter)
# print("---------Wrote Signatures Coefficient into a file----------")
#
# visualize.generateFrequencyInformation(X,dict_SignatureHR,uhidSepsisCase,uhidNoSepsisCase)
# plt.show()
#
# #Now plot the signature for both sepsis and non-sepsis categories
# visualize.plotSepsisAndNoSepsisSignatureCoefficient(debugCode, numberOfPatients, uhidSepsisCase,uhidNoSepsisCase,X,lengthofDataToProcess, windowLength, zippedUHIDTupleDict)
# print("---------Plot signature coefficients done----------")
#
# visualize.plotDataAndSignature(X,dict_SignatureHR,uhidSepsisCase,uhidNoSepsisCase,lengthofDataToProcess,windowLength)
#
#
#
# print("---------Modeling Data----------")
# model = modelhelper.executeModelWithOneLeaveOut(dict_SignatureHR,y_final,listUHID,timeBlocksCounter,signatureLength)
#
#
