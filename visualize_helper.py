import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import scipy.signal
from matplotlib import pyplot as plt

def convertTo1DArrayTuple(inputTuple):
    hrArray =[]
    spo2Array=[]
    for x in inputTuple:
        #x contains spo2 and hr
        hrArray.append(x[0])
        spo2Array.append(x[1])
    return np.array(hrArray),np.array(spo2Array)

def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s)
        # pre-sorting of locals min based on relative position with respect to s_mid
        lmin = lmin[s[lmin] < s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid
        lmax = lmax[s[lmax] > s_mid]

    # global min of dmin-chunks of locals min
    lmin = lmin[[i + np.argmin(s[lmin[i:i + dmin]]) for i in range(0, len(lmin), dmin)]]
    # global max of dmax-chunks of locals max
    lmax = lmax[[i + np.argmax(s[lmax[i:i + dmax]]) for i in range(0, len(lmax), dmax)]]

    return lmin, lmax

def generateFrequencyInformation(X,dict_Signature,uhidSepsisCase,uhidNoSepsisCase):
    plt.figure(5)
    figure5, axis5 = plt.subplots(6, 1)
    figure5.suptitle('Figure 1- plot of sepsis and no-sepsis after convolution', fontsize=16)
    print('sepsis uhid=',uhidSepsisCase,' No Sepsis uhid=',uhidNoSepsisCase)
    sepsisPatient = X[uhidSepsisCase+'_'+'sepsis'][0]
    noSepsisPatient = X[uhidNoSepsisCase + '_' + 'nosepsis'][0]

    hrArray,spO2Array = convertTo1DArrayTuple(sepsisPatient)
    hrArrayNoSepsis, spO2ArrayNoSepsis = convertTo1DArrayTuple(noSepsisPatient)
    kernel_size = 30
    kernel = np.ones(kernel_size) / kernel_size
    data_convolvedHR = np.convolve(hrArray, kernel, mode='valid')
    data_convolvedSPO2 = np.convolve(spO2Array, kernel, mode='valid')

    data_convolvedHRNoSepsis = np.convolve(hrArrayNoSepsis, kernel, mode='valid')
    data_convolvedSPO2NoSepsis = np.convolve(spO2ArrayNoSepsis, kernel, mode='valid')
    # lminHR, lmaxHR = hl_envelopes_idx(np.array(hrArray))
    # lminSPO2, lmaxSPO2 = hl_envelopes_idx(np.array(spO2Array))


    # sepsisSpectrum = fft.fft(sepsisPatient)
    # noSepsisSpectrum = fft.fft(noSepsisPatient)
    # freqSepsis = fft.fftfreq(len(sepsisSpectrum))
    # freqNoSepsis = fft.fftfreq(len(noSepsisSpectrum))
    # # apply a 3-pole lowpass filter at 0.1x Nyquist frequency
    # b, a = scipy.signal.butter(3, [.01, .02], 'band')
    # filteredBandPass = scipy.signal.lfilter(b, a, sepsisPatient)

    axis5[0].set_title('Sepsis Patient')
    axis5[0].plot(np.array(sepsisPatient)[:,0])
    axis5[1].set_title('No Sepsis Patient')
    axis5[1].plot(np.array(noSepsisPatient)[:,0])
    axis5[2].set_title('Averaged/Convolved Sepsis HR')
    axis5[2].plot(data_convolvedHR)
    axis5[3].set_title('Averaged/Convolved Sepsis Sp02')
    axis5[3].plot(data_convolvedSPO2)
    axis5[4].set_title('Averaged/Convolved No Sepsis HR')
    axis5[4].plot(data_convolvedHRNoSepsis)
    axis5[5].set_title('Averaged/Convolved No Sepsis SpO2')
    axis5[5].plot(data_convolvedSPO2NoSepsis)

    return True

def plotSepsisAndNoSepsisSignatureCoefficient(debugCode, numberOfPatients, uhidSepsisCase,uhidNoSepsisCase,X,timeBlockSize, windowLength, zippedUHIDTupleDict):
    plt.figure(1)
    if (debugCode):
        figure1, axis1 = plt.subplots(2, 2)
    else:
        figure1, axis1 = plt.subplots(numberOfPatients, 3)
    debugSepsisPlot = False
    debugNoSepsisPlot = False
    debugCounter = 0
    figure1.suptitle('Figure 3- plot of sepsis and no-sepsis signature data', fontsize=16)
    sepsisPatient = X[uhidSepsisCase+'_'+'sepsis'][0]
    noSepsisPatient = X[uhidNoSepsisCase + '_' + 'nosepsis'][0]
    i=0
    counter =0
    while (i < timeBlockSize - 1):
        keySepsisValue = str(uhidSepsisCase) + '_' + str(counter)
        keyNoSepsisValue = str(uhidNoSepsisCase) + '_' + str(counter)
        sepsisPatientData = sepsisPatient[i:i+windowLength]
        nosepsisPatientData = noSepsisPatient[i:i+windowLength]
        two_dim_stream_physiological_sepsis = np.array(sepsisPatientData)
        two_dim_stream_physiological_nosepsis = np.array(nosepsisPatientData)
        hr_stream_physiological_sepsis = two_dim_stream_physiological_sepsis[:, 0]
        spo2_stream_physiological_sepsis = two_dim_stream_physiological_sepsis[:, 1]
        axis1[0, counter].set_title('Raw HR data')
        axis1[0, counter].plot(hr_stream_physiological_sepsis)
        axis1[0, counter].set_title('Raw SpO2 data')
        axis1[1, counter].plot(spo2_stream_physiological_sepsis)
        i = i + windowLength

    print('sepsis case uhid=', uhidSepsisCase, ' nosepsis uhid=', uhidNoSepsisCase)
    coefficientListNoSepsis = list(zippedUHIDTupleDict[uhidNoSepsisCase])
    coefficientListSepsis = list(zippedUHIDTupleDict[uhidSepsisCase])
    for eachCoefficent in coefficientListSepsis:
        eachCoefficentList = list(eachCoefficent)
        for coefficient in eachCoefficentList:
            print(coefficient)
            axis1[0, 1].plot(coefficient)
        axis1[0, 1].set_title('Coefficients of Signature')
        axis1[0, 1].set_xlabel('Depth is 3 and 15 elements')
        axis1[0, 1].set_ylabel('Coefficients')
    for eachCoefficent in coefficientListNoSepsis:
        eachCoefficentList = list(eachCoefficent)
        for coefficient in eachCoefficentList:
            print(coefficient)
            axis1[1, 1].plot(coefficient)
        axis1[1, 1].set_title('Coefficients of Signature')
        axis1[1, 1].set_xlabel('Depth is 3 and 15 elements')
        axis1[1, 1].set_ylabel('Coefficients')

def plotDataAndSignature(X,dict_SignatureHR,uhidSepsisCase,uhidNoSepsisCase,timeBlockSize, windowLength):
    print('sepsis uhid=',uhidSepsisCase,' No Sepsis uhid=',uhidNoSepsisCase)
    # In figure two plot time by time data and its corresponding signature
    plt.figure(2)
    timeBlocksCounter = int(timeBlockSize/windowLength)
    figure2, axis2 = plt.subplots(2, timeBlocksCounter)
    figure2.suptitle('Figure 2 (a)- block wise sepsis & signature content', fontsize=14)

    plt.figure(3)
    figure3, axis3 = plt.subplots(2, timeBlocksCounter)
    figure3.suptitle('Figure 2 (b)- block wise No sepsis & signature content', fontsize=14)

    sepsisPatient = X[uhidSepsisCase+'_'+'sepsis'][0]
    noSepsisPatient = X[uhidNoSepsisCase + '_' + 'nosepsis'][0]



    i=0
    counter =0
    plt.figure(4)
    figure4, axis4 = plt.subplots(2, 1)
    figure4.suptitle('Figure 00- plot of sepsis and no-sepsis total raw data as is', fontsize=16)
    axis4[0].plot(np.array(sepsisPatient)[:,[0,1]])
    axis4[0].set_ylabel('HR/SpO2 Continuous')
    axis4[1].plot(np.array(noSepsisPatient)[:,[0,1]])
    axis4[1].set_ylabel('HR/SpO2 Continuous')


    while (i < timeBlockSize - 1):
        keySepsisValue = str(uhidSepsisCase) + '_' + str(counter)
        keyNoSepsisValue = str(uhidNoSepsisCase) + '_' + str(counter)
        sepsisPatientData = sepsisPatient[i:i+windowLength]
        nosepsisPatientData = noSepsisPatient[i:i+windowLength]
        signatureDataSepsis = dict_SignatureHR[keySepsisValue][0]
        signatureDataNoSepsis = dict_SignatureHR[keyNoSepsisValue][0]
        #plot HR as the signature is HR
        two_dim_stream_physiological_HR_sepsis = np.array(sepsisPatientData)[:, 0]
        two_dim_stream_physiological_HR_nosepsis = np.array(nosepsisPatientData)[:, 0]

        axis2[0, counter].plot(two_dim_stream_physiological_HR_sepsis)
        axis2[1, counter].plot(signatureDataSepsis)
        #only for the first chart show the y label but hide for other 120 min charts to give continuous feeling to user
        if(counter==0):
            axis2[0, counter].set_ylabel('HR/SpO2 Amplitude')
            axis2[1, counter].set_ylabel('Signature Coefficients')
        else:
            # Hide X and Y axes label marks
            axis2[0, counter].xaxis.set_tick_params(labelbottom=False)
            axis2[0, counter].yaxis.set_tick_params(labelleft=False)
            # Hide X and Y axes tick marks
            axis2[0, counter].set_xticks([])
            axis2[0, counter].set_yticks([])
            # Hide X and Y axes label marks
            axis2[1, counter].xaxis.set_tick_params(labelbottom=False)
            axis2[1, counter].yaxis.set_tick_params(labelleft=False)
            # Hide X and Y axes tick marks
            axis2[1, counter].set_xticks([])
            axis2[1, counter].set_yticks([])

        axis3[0, counter].plot(two_dim_stream_physiological_HR_nosepsis)
        axis3[1, counter].plot(signatureDataNoSepsis)
        if (counter == 0):
            axis3[0, counter].set_ylabel('HR/SpO2 Amplitude')
            axis3[1, counter].set_ylabel('Signature Coefficients')
        else:
            # Hide X and Y axes label marks
            axis3[0, counter].xaxis.set_tick_params(labelbottom=False)
            axis3[0, counter].yaxis.set_tick_params(labelleft=False)
            # Hide X and Y axes tick marks
            axis3[0, counter].set_xticks([])
            axis3[0, counter].set_yticks([])
            # Hide X and Y axes label marks
            axis3[1, counter].xaxis.set_tick_params(labelbottom=False)
            axis3[1, counter].yaxis.set_tick_params(labelleft=False)
            # Hide X and Y axes tick marks
            axis3[1, counter].set_xticks([])
            axis3[1, counter].set_yticks([])
        i = i + windowLength
        counter = counter + 1
    print('This method prints data and signature in second figure')