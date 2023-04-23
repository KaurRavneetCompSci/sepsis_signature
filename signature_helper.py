import esig.tosig as ts
import numpy as np
from collections import defaultdict
import csv
from sklearn.preprocessing import normalize

def generateSignature(X,Y, depthSignature, windowLength,lengthofDataToProcess):
    y_final = []
    dict_SignatureHR = defaultdict(list)
    dict_SignatureSpo2 = defaultdict(list)
    # debugging related flags
    debugSepsisPlot = False
    debugNoSepsisPlot = False
    uhidSepsisCase = ''
    uhidNoSepsisCase = ''
    debugCounter = 0
    signatureLength = 0
    indexPatient= 0
    timeBlocksCounter = 0
    listUHID = []
    for uhidKey in X:
        i = 0
        timeBlocksCounter = 0
        splitResultsUHID = uhidKey.split('_')
        uhid = splitResultsUHID[0]
        patientCaseType = splitResultsUHID[1]
        #aim is to do one for sepsis and one for nosepsis in debug mode
        eachPatient = X[uhidKey][0]
        # print(len(eachPatient))
        listUHID.append(uhid)
        if (patientCaseType == 'nosepsis'):
            uhidNoSepsisCase = uhid
            debugNoSepsisPlot = True
        while(i < lengthofDataToProcess):
            eachPatientData = eachPatient[i:i+windowLength]
            #uhid is the first column - remove it
            eachPatientData = eachPatientData[1:]
            two_dim_stream_physiological = np.array(eachPatientData)
            y_final.append(Y[indexPatient])
            hr_stream_physiological = two_dim_stream_physiological[:,[0,2]]
            spo2_stream_physiological = two_dim_stream_physiological[:, [1,2]]
            keyValue = str(uhid)+'_'+str(timeBlocksCounter)
            # print(str(signatureLength))
            #Signature size of input array of dimension 3 is of length 40. for dimension 2 it is 15
            hr_stream_physiological_nm = normalize(hr_stream_physiological, axis=0, norm='max')
            spo2_stream_physiological_nm = normalize(spo2_stream_physiological, axis=0, norm='max')

            signatureOutputHR = ts.stream2sig(hr_stream_physiological_nm, depthSignature)
            signatureOutputSpo2 = ts.stream2sig(spo2_stream_physiological_nm, depthSignature)
            dict_SignatureHR[keyValue].append(list(signatureOutputHR))
            dict_SignatureSpo2[keyValue].append(list(signatureOutputSpo2))
            i = i + windowLength
            timeBlocksCounter = timeBlocksCounter +1
        indexPatient = indexPatient+1
        if (patientCaseType == 'sepsis' and not(debugSepsisPlot)):
            # Set title and other properties of plot
            uhidSepsisCase = uhid
            debugCounter = debugCounter + 1
            debugSepsisPlot = True
        # if((debugSepsisPlot and debugNoSepsisPlot)):
        #     break
    #two_dim_stream_physiological = np.random.random(size=(121,2))
    # zippedUHIDTuple = list(repeat([], numberOfPatients))
    return dict_SignatureHR,dict_SignatureSpo2, listUHID, uhidSepsisCase, uhidNoSepsisCase,y_final


def concatenateSignatureCoefficients(dict_Signature,listUHID,timeBlocksCounter):
    zippedUHIDTupleDict = defaultdict(list)
    # now iterate the dictionary to sequence each coefficient and then plot
    # print('counter=',counter)
    patientCounter = 0
    for eachUHID in listUHID:
        i = 0
        eachUHIDData = []
        while (i < timeBlocksCounter):
            keyValue = str(eachUHID) + '_' + str(i)
            eachUHIDData.append(dict_Signature[keyValue][0])
            i = i + 1
        zippedUHIDTupleDict[eachUHID].append(zip(*eachUHIDData))
        patientCounter = patientCounter + 1
    patientCounter = 0
    return zippedUHIDTupleDict

def writeSignatureCoefficientsIntoCSV(dataFilePath, dict_Signature,listUHID,timeBlocksCounter):
    fileOpen = open(dataFilePath+'signature.csv', 'w')
    # create the csv writer
    writer = csv.writer(fileOpen)
    # now iterate the dictionary to write each coefficient
    patientCounter = 0
    for eachUHID in listUHID:
        i = 0
        eachUHIDData = []
        while (i < timeBlocksCounter):
            keyValue = str(eachUHID) + '_' + str(i)
            writer.writerow(dict_Signature[keyValue][0])
            i = i + 1
        patientCounter = patientCounter + 1
    patientCounter = 0
    # close the file
    fileOpen.close()
    return True