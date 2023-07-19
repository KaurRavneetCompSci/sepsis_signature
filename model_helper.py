from keras.layers import  Dense
from keras.layers import LSTM
from keras.models import Sequential, Model
# from keras.optimizers import adam_v2
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def create_model(n_features,n_steps):
    model = Sequential()
    inputNumberNeuron = 512
    multiplEachLayer = (1)
    n_dropout = [0.2]
    hiddenLayerNeuron1 = int(multiplEachLayer * inputNumberNeuron)
    dropout = 0.2
    model = Sequential()
    #Simple model
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    #our model
    # model.add(Bidirectional(LSTM(inputNumberNeuron, activation='tanh',return_sequences=True,dropout=dropout), input_shape=(n_steps, n_features)))
    model.add(Dense(1, activation="sigmoid"))
    model.summary()
    model.compile(optimizer=adam_v2.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_evaluate(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
 #print model characterstics
    modeloutput = model.predict(x_train)
    cutoff = 0.5
    y_output_ontraining = np.where(modeloutput > cutoff, 1, 0)
    print('-------------TRAINING RESULT START------')
    cm = confusion_matrix(y_train, y_output_ontraining)
    ac = accuracy_score(y_train, y_output_ontraining)
    print(cm)
    print(ac)
    print('-------------TRAINING RESULT END------')

    return model.predict(x_test)

def executeModelWithOneLeaveOut(dict_SignatureHR,y_final,listUHID,timeBlocksCounter,numberofCoefficients):
    #input data to the model is list of merged data for all patients
    list_Signature = []
    patientCounter = 0
    for eachUHID in listUHID:
        i = 0
        while (i < timeBlocksCounter):
            keyValue = str(eachUHID) + '_' + str(i)
            list_Signature.append(dict_SignatureHR[keyValue][0])
            i = i + 1
        patientCounter = patientCounter + 1
    #Model for time series modeliing
    #y is sepsis or no sepsis
    n_features = 1
    # list_Signature = list_Signature.reshape((list_Signature.shape[0], list_Signature.shape[1], n_features))
    idx = 0
    scores = np.zeros(10)
    kFold = KFold(n_splits=patientCounter)
    X = np.reshape(list_Signature, (patientCounter*timeBlocksCounter,numberofCoefficients, n_features))
    y_final = np.reshape(y_final,(patientCounter*timeBlocksCounter, n_features))
    y_final = np.array(y_final)
    for train, test in kFold.split(X, y_final):
        model = create_model(n_features,numberofCoefficients)
        prediction = train_evaluate(model, X[train], y_final[train], X[test], y_final[test])
        cutoff = 0.5
        y_prediction = np.where(prediction > cutoff, 1, 0)
        cm = confusion_matrix(y_final[test], y_prediction)
        ac = accuracy_score(y_final[test], y_prediction)
        print(cm)
        print(ac)
        idx += 1
