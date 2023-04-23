import matplotlib.pyplot as plt
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
depthSignature = 5
signatureLength = ts.sigdim(2, depthSignature)

#uhid plus the time block is the key and signature of each block of time i.e. 120 seconds is the value
#three days of data for each patient
lengthofDataToProcess = 2*60
timeBlocksCounter = int(lengthofDataToProcess/windowLength)
y_final = []
debugCode = True
print("---------Preparing Data----------")
dataFilePath = '/Users/harpreetsingh/Harpreet/Ravneet/Sepsis_Signature/generalised-signature-method/'
dataPreparation = prepare.DataPreparation(dataFilePath)
#below method return dictionary object with uhid_sepsis or uhid_nosepsis as key and list of data as value
X,Y = dataPreparation.prepareData('sepsis_nosepsis.csv',lengthofDataToProcess)
print('-------------Data Preparation Done------------')
numberOfPatients  = len(X)
dict_SignatureHR, dict_SignatureSpO2, listUHID, uhidSepsisCase, uhidNoSepsisCase,y_final = signature.generateSignature(X,Y, depthSignature, windowLength,lengthofDataToProcess)
print("---------Signatures Done----------")

zippedUHIDTupleDict = signature.concatenateSignatureCoefficients(dict_SignatureHR,listUHID,timeBlocksCounter)
print("---------Signatures Coefficient Concatenation Done----------")

signature.writeSignatureCoefficientsIntoCSV(dataFilePath, dict_SignatureHR,listUHID,timeBlocksCounter)
print("---------Wrote Signatures Coefficient into a file----------")

#Now plot the signature for both sepsis and non-sepsis categories
visualize.plotSepsisAndNoSepsisSignatureCoefficient(debugCode, numberOfPatients, uhidSepsisCase,uhidNoSepsisCase,X,lengthofDataToProcess, windowLength, zippedUHIDTupleDict)
print("---------Plot signature coefficients done----------")

visualize.plotDataAndSignature(X,dict_SignatureHR,uhidSepsisCase,uhidNoSepsisCase,lengthofDataToProcess,windowLength)

visualize.generateFrequencyInformation(X,dict_SignatureHR,uhidSepsisCase,uhidNoSepsisCase)
plt.show()

print("---------Modeling Data----------")
model = modelhelper.executeModelWithOneLeaveOut(dict_SignatureHR,y_final,listUHID,timeBlocksCounter,signatureLength)


