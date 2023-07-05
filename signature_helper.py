import esig.tosig as ts
import numpy as np
from collections import defaultdict
import csv
from sklearn.preprocessing import normalize
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse
from tsaug.visualization import plot
import similaritymeasures
import matplotlib.pyplot as plt


# This method takes the patient list of sepsis and no-sepsis cases. Each record is augmented using
# time series augmentation -tsaug. one record becomes two.
def extractFirst(lst):
    return [item[0] for item in lst]


def generateSignature(X, Y, depthSignature, windowLength, lengthofDataToProcess, dataPreparation, dataFilePath):
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
    indexPatient = 0
    timeBlocksCounter = 0
    listUHID = []
    biggerListUHID = []
    XAugHR = defaultdict(list)
    XAugSpO2 = defaultdict(list)
    my_augmenter = (TimeWarp() * 2  # random time warping 5 times in parallel
                    #                 + Crop(size=300)  # random crop subsequences with length 300
                    #                + Quantize(n_levels=[10, 20, 30])  # random quantize to 10-, 20-, or 30- level sets
                    + Drift(
                max_drift=(0.1, 0.5)) @ 0.8  # with 80% probability, random drift the signal up to 10% - 50%
                    #   + Reverse() @ 0.5  # with 50% probability, reverse the sequence
                    )
    for uhidKey in X:
        print(uhidKey)
        i = 0
        timeBlocksCounter = 0
        splitResultsUHID = uhidKey.split('_')
        uhid = splitResultsUHID[0]
        patientCaseType = splitResultsUHID[1]
        # aim is to do one for sepsis and one for nosepsis in debug mode
        eachPatient = X[uhidKey][0]
        # print(len(eachPatient))
        listUHID.append(uhid)
        biggerListUHID.append(uhidKey)
        # add key for augmented uhid generated from this patient
        listUHID.append(uhid + '_aug')
        if (patientCaseType == 'nosepsis'):
            uhidNoSepsisCase = uhid
            debugNoSepsisPlot = True
        augKeyValue = uhidKey + '_aug'
        patientAugDataHR = []
        patientAugDataSpO2 = []
        while (i < lengthofDataToProcess):
            eachPatientData = eachPatient[i:i + windowLength]
            two_dim_stream_physiological = np.array(eachPatientData)
            y_final.append(Y[indexPatient])
            # put timeseries as first column and physiological signal as second column
            hr_stream_physiological = two_dim_stream_physiological[:, [2, 0]]
            spo2_stream_physiological = two_dim_stream_physiological[:, [2, 1]]
            keyValue = str(uhid) + '_' + str(timeBlocksCounter)
            # print(str(signatureLength))
            # Signature size of input array of dimension 3 is of length 40. for dimension 2 it is 15
            hr_stream_physiological_nm = normalize(hr_stream_physiological, axis=0, norm='max')
            spo2_stream_physiological_nm = normalize(spo2_stream_physiological, axis=0, norm='max')

            aug_hr = my_augmenter.augment(hr_stream_physiological)
            aug_spo2 = my_augmenter.augment(spo2_stream_physiological)

            aug_hr_nm = normalize(aug_hr, axis=0, norm='max')
            aug_spo2_nm = normalize(aug_spo2, axis=0, norm='max')

            # #Do Data Warping and augmentation
            # print("Output:")
            # print(aug_hr)

            keyValue_aug = str(uhid) + '_' + patientCaseType + '_aug_' + str(timeBlocksCounter)

            # Now do the signature
            signatureOutputHR = ts.stream2sig(hr_stream_physiological_nm, depthSignature)
            signatureOutputSpo2 = ts.stream2sig(spo2_stream_physiological_nm, depthSignature)

            # Now do the signature of augmented signal
            signatureOutputHR_aug = ts.stream2sig(aug_hr_nm, depthSignature)
            signatureOutputSpo2_aug = ts.stream2sig(aug_spo2_nm, depthSignature)

            tupleHR = list(zip(aug_hr[:, 1], aug_hr[:, 0]))
            tupleSpO2 = list(zip(aug_spo2[:, 1], aug_spo2[:, 0]))
            # Update the data in list - both original and augmented
            # Populate HR augmented data
            if patientAugDataHR:
                patientAugDataHR.extend(tupleHR)
            else:
                patientAugDataHR = tupleHR

            # Populate SpO2 augmented data
            if patientAugDataSpO2:
                patientAugDataSpO2.extend(tupleSpO2)
            else:
                patientAugDataSpO2 = tupleSpO2

            dict_SignatureHR[keyValue].append(list(signatureOutputHR))
            dict_SignatureSpo2[keyValue].append(list(signatureOutputSpo2))

            dict_SignatureHR[keyValue_aug].append(list(signatureOutputHR_aug))
            dict_SignatureSpo2[keyValue_aug].append(list(signatureOutputSpo2_aug))

            i = i + windowLength
            timeBlocksCounter = timeBlocksCounter + 1

        actualUHID = uhid.split(".")[0]
        k = 0
        eachPatientDataAugmented = patientAugDataHR[k + 1:k + (2 * lengthofDataToProcess):2]
        if ('nosepsis' in uhidKey):
            folderName = "/NoSepsis_Cases/"
            typeOfCase = "Discharge"
            conditionCase = "(dischargestatus = 'Discharge')"
            fileName, preparedData = dataPreparation.read_prepare_data_auto(actualUHID, typeOfCase, conditionCase,
                                                                            folderName)

            finalList = extractFirst(eachPatientDataAugmented)
            list_inter = [None] * (len(preparedData) - len(finalList))
            finalList.extend(list_inter)
            preparedData['HR_aug'] = finalList
            preparedData.to_csv(dataFilePath + folderName + actualUHID + "/" + actualUHID + "_aug.csv")

        else:
            folderName = "/Sepsis_Cases/"
            conditionCase = "(dischargestatus = 'Death' or dischargestatus = 'LAMA' or dischargestatus = 'Sepsis' )"
            typeOfCase = "Death"
            fileName, preparedData = dataPreparation.read_prepare_data_auto(actualUHID, typeOfCase, conditionCase,
                                                                            folderName)

            sepsis = preparedData[preparedData['sepsis'] == 1]
            finalList = extractFirst(eachPatientDataAugmented)
            sepsisIDX = (sepsis['Unnamed: 0'].iloc[0])
            list_inter = [None] * (sepsisIDX - len(finalList))
            list_inter.extend(finalList)
            list_inter_2 = [None] * (len(preparedData) - len(list_inter))
            list_inter.extend(list_inter_2)
            preparedData['HR_aug'] = list_inter
            preparedData.to_csv(dataFilePath + folderName + actualUHID + "/" + actualUHID + "_aug.csv")

        XAugHR[augKeyValue].append(patientAugDataHR)
        XAugSpO2[augKeyValue].append(patientAugDataSpO2)
        indexPatient = indexPatient + 1
        if (patientCaseType == 'sepsis' and not (debugSepsisPlot)):
            # Set title and other properties of plot
            uhidSepsisCase = uhid
            debugCounter = debugCounter + 1
            debugSepsisPlot = True
        # if((debugSepsisPlot and debugNoSepsisPlot)):
        #     break
    # two_dim_stream_physiological = np.random.random(size=(121,2))
    # zippedUHIDTuple = list(repeat([], numberOfPatients))
    return dict_SignatureHR, dict_SignatureSpo2, listUHID, uhidSepsisCase, uhidNoSepsisCase, y_final, biggerListUHID, XAugHR, XAugSpO2


def concatenateSignatureCoefficients(dict_Signature, listUHID, timeBlocksCounter):
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


def writeSignatureCoefficientsIntoCSV(dataFilePath, dict_Signature, listUHID, timeBlocksCounter):
    fileOpen = open(dataFilePath + 'signature.csv', 'w')
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


def similarityBetweenSignatureCoefficients(dict_Signature, listUHID, timeBlocksCounter, n):
    # now iterate the dictionary to iterate each coefficient
    # now iterate the dictionary to sequence each coefficient and then plot
    # print('counter=',counter)
    patientCounter = 0
    m = len(listUHID)
    # excluding the first number from signature as thats always 1
    sepsis_data = np.empty(shape=[m * timeBlocksCounter * (n - 1)])
    nosepsis_data = np.empty(shape=[m * timeBlocksCounter * (n - 1)])
    for eachUHID in listUHID:
        splitResultsUHID = eachUHID.split('_')
        uhid = splitResultsUHID[0]
        patientCaseType = splitResultsUHID[1]
        i = 0
        eachUHIDData = np.empty(shape=[timeBlocksCounter * (n - 1)])
        while (i < timeBlocksCounter):
            keyValue = str(uhid) + '_' + str(i)
            signatureData = np.array(dict_Signature[keyValue][0])
            # skip the first coefficient as thats always 1
            signatureData = signatureData[1:]
            np.append(eachUHIDData, signatureData)
            i = i + 1
        if (patientCaseType == 'nosepsis'):
            np.append(nosepsis_data, eachUHIDData)
        if (patientCaseType == 'sepsis'):
            np.append(sepsis_data, eachUHIDData)
        patientCounter = patientCounter + 1
    area = similaritymeasures.area_between_two_curves(sepsis_data, nosepsis_data)
    mse = similaritymeasures.mse(sepsis_data, nosepsis_data)
    # mean absolute error
    mae = similaritymeasures.mae(sepsis_data, nosepsis_data)
    print(area, mae, mse)
    return sepsis_data, nosepsis_data