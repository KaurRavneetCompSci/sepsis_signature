import numpy as np
import similaritymeasures
import matplotlib.pyplot as plt



def similarityBetweenSignatureCoefficients(plotCounter,dict_Signature,listUHID,timeBlocksCounter,n):
    plt.show()
    plt.figure(plotCounter)

    # now iterate the dictionary to iterate each coefficient
    # now iterate the dictionary to sequence each coefficient and then plot
    # print('counter=',counter)
    patientCounter = 0
    m = len(listUHID)
    #excluding the first number from signature as thats always 1
    sepsis_data = np.empty(shape=m*timeBlocksCounter*(n-1))
    nosepsis_data = np.empty(shape=m*timeBlocksCounter*(n-1))
    patientCounterList = np.arange(0, m*timeBlocksCounter*(n-1), 1, dtype=int)
    sepsis_data_print = np.zeros((m*timeBlocksCounter*(n-1), 2))
    no_sepsis_data_print = np.zeros((m * timeBlocksCounter *(n - 1), 2))
    counterSepsis = 0
    counterNoSepsis = 1
    for eachUHID in listUHID:
        splitResultsUHID = eachUHID.split('_')
        uhid = splitResultsUHID[0]
        patientCaseType = splitResultsUHID[1]
        i = 0
        eachUHIDData = np.empty(shape=[timeBlocksCounter*(n-1)])
        while (i < timeBlocksCounter):
            keyValue = str(uhid) + '_' + str(i)
            signatureData = np.array(dict_Signature[keyValue][0])
            #skip the first coefficient as thats always 1
            signatureData = signatureData[1:]
            np.append(eachUHIDData,signatureData)
            i = i + 1
        if (patientCaseType == 'nosepsis'):
            np.append(nosepsis_data, eachUHIDData)
        if (patientCaseType == 'sepsis'):
            np.append(sepsis_data, eachUHIDData)
        patientCounter = patientCounter + 1
    sepsis_data_print[:, 0] = patientCounterList
    sepsis_data_print[:, 1] = sepsis_data
    no_sepsis_data_print[:, 0] = patientCounterList
    no_sepsis_data_print[:, 1] = nosepsis_data
    area = similaritymeasures.area_between_two_curves(sepsis_data_print, no_sepsis_data_print)
    mse = similaritymeasures.mse(sepsis_data_print, no_sepsis_data_print)
    # mean absolute error
    mae = similaritymeasures.mae(sepsis_data_print, no_sepsis_data_print)
    print( area, mae, mse)
    plt.figure(plotCounter)
    plt.plot(sepsis_data_print[:, 0], sepsis_data_print[:, 1])
    plt.plot(no_sepsis_data_print[:, 0], no_sepsis_data_print[:, 1])
    plt.show()