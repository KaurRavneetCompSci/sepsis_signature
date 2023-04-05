import matplotlib.pyplot as plt
import data_preparation as prepare
import visualize_helper as visualize
import signature_helper as signature
from collections import defaultdict
from cycler import cycler
import matplotlib as mpl

windowLength = 120
mpl.rcParams['axes.prop_cycle'] = cycler(color=['r', 'g', 'r', 'g'])
mpl.rcParams['lines.linewidth'] = 2
depthSignature = 3
#uhid plus the time block is the key and signature of each block of time i.e. 120 seconds is the value
#three days of data for each patient
lengthofDataToProcess = 3*24*60
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
dict_Signature,listUHID, uhidSepsisCase, uhidNoSepsisCase = signature.generateSignature(X,depthSignature, windowLength)
zippedUHIDTupleDict = signature.concatenateSignatureCoefficients(dict_Signature,listUHID,timeBlocksCounter)
signature.writeSignatureCoefficientsIntoCSV(dataFilePath, dict_Signature,listUHID,timeBlocksCounter)
#Now plot the signature for both sepsis and non-sepsis categories
visualize.plotSepsisAndNoSepsisSignatureCoefficient(debugCode, numberOfPatients, uhidSepsisCase,uhidNoSepsisCase,X,timeBlocksCounter, windowLength, zippedUHIDTupleDict)
visualize.plotDataAndSignature(X,dict_Signature,uhidSepsisCase,uhidNoSepsisCase,lengthofDataToProcess,windowLength)

visualize.generateFrequencyInformation(X,dict_Signature,uhidSepsisCase,uhidNoSepsisCase)
plt.show()

# #Model for time series modeliing
# model = Sequential()
# inputNumberNeuron = 512
# multiplEachLayer = (1)
# n_steps = 15
# n_dropout = [0.2]
# n_features = 1
# hiddenLayerNeuron1 = int(multiplEachLayer * inputNumberNeuron)
# dropout = 0.2
# model = Sequential()
# #Simple model
# model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
# #our model
# # model.add(Bidirectional(LSTM(inputNumberNeuron, activation='tanh',return_sequences=True,dropout=dropout), input_shape=(n_steps, n_features)))
# model.add(Dense(1, activation="sigmoid"))
# model.summary()
# model.compile(optimizer=adam_v2.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
# #y is sepsis or no sepsis
#
# list_Signature = np.array(list_Signature)
# print(list_Signature.shape)
# list_Signature = list_Signature.reshape((list_Signature.shape[0], list_Signature.shape[1], n_features))
# print(list_Signature.shape)
# y_final = np.array(y_final)
# print(y_final.shape)
# model.fit(list_Signature, y_final, epochs=200, verbose=0)

