import numpy as np

from tensorflow import keras
from keras import layers

import linecache
from collections import defaultdict
import pandas as pd


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_model(
        input_shape,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_units,
        dropout=0,
        mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

def read_prepare_data_auto(filePath, patientCaseUHID, caseType, conditionCase, folderName):
    try:
        # path = os.getcwd()
        path = filePath
        seperator = '_'
        filePath = path + folderName + str(patientCaseUHID) + "/"
        fileName = filePath + caseType + seperator + str(
            patientCaseUHID) + seperator + 'intermediate_checkpoint_new_5.csv'
        # print('fileName=>'+fileName)
        preparedData = pd.read_csv(fileName, low_memory=False)
        return fileName, preparedData
    except IOError as e:
        print('No data for this uhid in data preparation', e)
    except Exception as e:
        print('Exception in read_prepare_data', e)
        PrintException()
def prepareData(filePath, filename, dataRead):
    finalSetDS = pd.read_csv(filePath + filename, low_memory=False)
    # print('Length of balanced dataset', len(finalSetDS))
    preparedData = pd.DataFrame()
    labels_list = []
    i = 1
    typeOfCase = ""
    sepsisCase = 1
    noSepsisCase = 0
    dict_Signature = {}
    for row in finalSetDS.itertuples():
        try:
            # print(i,'---->',getattr(row, 'uhid'), getattr(row, 'dischargestatus'))
            i = i + 1
            patientCaseUHID = getattr(row, 'uhid')
            # print('patientCaseUHID', patientCaseUHID)
            caseType = getattr(row, 'dischargestatus')
            # print('caseType',caseType,type(caseType))
            if caseType == sepsisCase:
                folderName = "/Sepsis_Cases/"
                conditionCase = "(dischargestatus = 'Death' or dischargestatus = 'LAMA' or dischargestatus = 'Sepsis' )"
                typeOfCase = "Death"
            elif caseType == noSepsisCase:
                folderName = "/NoSepsis_Cases/"
                conditionCase = "(dischargestatus = 'Discharge')"
                typeOfCase = "Discharge"
            # print('patientCaseUHID',patientCaseUHID,'caseType',typeOfCase,'conditionCase',conditionCase,'folderName',folderName)
            # uncomment below to generate data first time
            # fileName,uhidDataSet = prepare_data(con,patientCaseUHID,typeOfCase,conditionCase,folderName)
            # uncomment below in case csv data is already generated and now lstm needs to be executed
            patientCaseUHID = str(patientCaseUHID)
            fileName, preparedData = read_prepare_data_auto(filePath, patientCaseUHID, typeOfCase, conditionCase,
                                                            folderName)
            if caseType == sepsisCase:
                preparedData['sepsis'].replace(to_replace=0.0, method='bfill', inplace=True, limit=dataRead)
                sepsis = preparedData[preparedData['sepsis'] == 1]
                sepsis = sepsis[['uhid', 'heartrate', 'spo2']]
                sepsis = sepsis.fillna(method='ffill')
                sepsis = sepsis.astype('float')
                labels_list.append(1.0)
                data_list = []
                # print(len(sepsis))
                uhidKey = []
                counterData = 0
                for index, row in sepsis.iterrows():
                    uhidKey = row['uhid']

                    counterData = counterData + 1
                    data_list.append(row['heartrate'])
                uhidCompositeKey = str(uhidKey) + '_sepsis'
                dict_Signature[uhidCompositeKey] = (data_list)
            else:
                preparedData = preparedData[preparedData['heartrate'] >= 0]
                preparedData = preparedData[preparedData['spo2'] >= 0]
                preparedData = preparedData.head(dataRead + 1)
                sepsis = preparedData[['uhid', 'heartrate', 'spo2']]
                sepsis = sepsis.astype('float')
                labels_list.append(0.0)
                data_list = []
                # print(len(sepsis))
                uhidKey = []
                counterData = 0
                for index, row in sepsis.iterrows():
                    uhidKey = row['uhid']
                    data_list.append(row['heartrate'])
                    counterData = counterData + 1
                uhidCompositeKey = str(uhidKey) + '_nosepsis'
                dict_Signature[uhidCompositeKey] = (data_list)
        except Exception as e:
            print('Exception in prediction_data_death_discharge', e)
    print(len(dict_Signature))
    # print(len(labels_list))
    X = (dict_Signature)
    Y = (labels_list)
    return X, Y


def PrintException():
    import sys
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)

from collections import defaultdict
import data_preparation as prepare

windowLength = 10

# depthSignature = 2
# signatureLength = ts.sigdim(2, depthSignature)

#uhid plus the time block is the key and signature of each block of time i.e. 120 seconds is the value
#three days of data for each patient
lengthofDataToProcess = 2*60
timeBlocksCounter = int(lengthofDataToProcess/windowLength)
y_final = []
debugCode = True
print("---------Preparing Data----------")
dataFilePath = '/Users/ravneetkaur/SignatoryProject/'
# dataPreparation = prepare.DataPreparation(dataFilePath)
#below method return dictionary object with uhid_sepsis or uhid_nosepsis as key and list of data as value
X,Y = prepareData(dataFilePath, 'sepsis_nosepsis.csv',lengthofDataToProcess)

X,Y = prepareData(dataFilePath, 'sepsis_nosepsis.csv',lengthofDataToProcess)

import numpy
x_list = []
for data in X:
    if(len(X[data]) == 242):
        list = X[data]
        x_list.append(list[0:121])
        x_list.append(list[121:])
        Y.insert(1,1)
    else:
        x_list.append(X[data])

x_list = numpy.array(x_list)
y_list = numpy.array(Y)


# print((x_train))
print((y_list))

print('-------------Data Preparation Done------------')
print(x_list.shape, " ", y_list.shape)

x_list = x_list.reshape((x_list.shape[0], x_list.shape[1], 1))
# x_test = x_test.reshape((x_test.shape[0], Y.shape[1], 1))
print(x_list.shape, " ", y_list.shape)

n_classes = len(np.unique(y_list))

iterations = 1
from sklearn.model_selection import train_test_split
bestAccuracy = 0
accuracy = 0
iterations = 5
for i in range(iterations):
    print("counter" , i)
    x_train, x_test, y_train, y_test = train_test_split(x_list, y_list, train_size=0.80)
    input_shape = x_train.shape[1:]

    model = build_model(
        input_shape,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["sparse_categorical_accuracy"],
    )

    callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

    model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=64,
        callbacks=callbacks,verbose = 0
    )

    s = model.evaluate(x_test, y_test, verbose=1)
    if(s[1] > bestAccuracy):
        bestAccuracy = s[1]
    accuracy = accuracy + s[1]
print("Best Accuracy = ", bestAccuracy)
print("Average Accuracy = ", accuracy / iterations)