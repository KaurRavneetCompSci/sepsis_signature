import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tsaug.visualization import plot


# def createDataSetForTranformer():
#     actualUHID = uhid.split(".")[0]
#     k = 0
#     eachPatientDataAugmented = patientAugDataHR[k + 1:k + (2 * lengthofDataToProcess):2]
#     eachPatientDataAugmented = extractFirst(eachPatientDataAugmented)
#
#     if ('nosepsis' in uhidKey):
#         folderName = "/NoSepsis_Cases/"
#         typeOfCase = "Discharge"
#         conditionCase = "(dischargestatus = 'Discharge')"
#         fileName, preparedData = dataPreparation.read_prepare_data_auto(actualUHID, typeOfCase, conditionCase,
#                                                                         folderName)
#
#         finalList = eachPatientDataAugmented
#         list_inter = [None] * (len(preparedData) - len(finalList))
#         finalList.extend(list_inter)
#         preparedData['HR_aug'] = finalList
#         preparedData.to_csv(dataFilePath + folderName + actualUHID + "/" + actualUHID + "_aug.csv")
#         uhidStr = actualUHID + '.0_nosepsis'
#         data = X_list[uhidStr]
#         final_data_for_transformer_Y.append(0)
#         final_data_for_transformer_Y.append(0)
#     else:
#         folderName = "/Sepsis_Cases/"
#         conditionCase = "(dischargestatus = 'Death' or dischargestatus = 'LAMA' or dischargestatus = 'Sepsis' )"
#         typeOfCase = "Death"
#         fileName, preparedData = dataPreparation.read_prepare_data_auto(actualUHID, typeOfCase, conditionCase,
#                                                                         folderName)
#
#         sepsis = preparedData[preparedData['sepsis'] == 1]
#         finalList = eachPatientDataAugmented
#         sepsisIDX = (sepsis['Unnamed: 0'].iloc[0])
#         list_inter = [None] * (sepsisIDX - len(finalList))
#         list_inter.extend(finalList)
#         list_inter_2 = [None] * (len(preparedData) - len(list_inter))
#         list_inter.extend(list_inter_2)
#         preparedData['HR_aug'] = list_inter
#         preparedData.to_csv(dataFilePath + folderName + actualUHID + "/" + actualUHID + "_aug.csv")
#         uhidStr = actualUHID + '.0_sepsis'
#         data = X_list[uhidStr]
#         final_data_for_transformer_Y.append(1)
#         final_data_for_transformer_Y.append(1)
#     list_inter_2 = []
#     for m in range(lengthofDataToProcess):
#         inter_list = [data[0][m]]
#         list_inter_2.append(inter_list)
#     final_data_for_transformer.append(list_inter_2)
#     list_inter_2 = []
#     for m in range(lengthofDataToProcess):
#         inter_list = [finalList[m]]
#         list_inter_2.append(inter_list)
#     final_data_for_transformer.append(list_inter_2)
#     XAugHR[augKeyValue].append(patientAugDataHR)
#     XAugSpO2[augKeyValue].append(patientAugDataSpO2)
#     indexPatient = indexPatient + 1
#     if (patientCaseType == 'sepsis' and not (debugSepsisPlot)):
#         # Set title and other properties of plot
#         uhidSepsisCase = uhid
#         debugCounter = debugCounter + 1
#         debugSepsisPlot = True
#     x_list = numpy.array(final_data_for_transformer)
#     y_list = numpy.array(final_data_for_transformer_Y)
#
#     # print((x_train))
#     print((y_list))
#
#     print('-------------Data Preparation Done------------')
#     print(x_list.shape, " ", y_list.shape)
#
#     x_list = x_list.reshape((x_list.shape[0], x_list.shape[1], 1))
#     # x_test = x_test.reshape((x_test.shape[0], Y.shape[1], 1))
#     print(x_list.shape, " ", y_list.shape)
#
#     n_classes = len(np.unique(y_list))
#     print(n_classes)
def runTransformer(x_list, y_list):
    from sklearn.model_selection import train_test_split
    bestAccuracy = 0
    accuracy = 0
    iterations = 50
    for i in range(iterations):
        print("counter", i)
        x_train, x_test, y_train, y_test = train_test_split(x_list, y_list, train_size=0.80)

        y_train = tf.one_hot(y_train, depth=2)
        y_test = tf.one_hot(y_test, depth=2)

        input_shape = x_train.shape[1:]

        model = build_model(
            input_shape,
            head_size=256,
            num_heads=4,
            ff_dim=2,
            num_transformer_blocks=2,
            mlp_units=[16],
            mlp_dropout=0.1,
            dropout=0.1,
        )

        model.compile(
            loss="binary_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            metrics=["accuracy"],
        )

        callbacks = [keras.callbacks.EarlyStopping(patience=10, monitor="accuracy", restore_best_weights=True)]

        model.fit(
            x_train,
            y_train,
            validation_split=0.2,
            epochs=10000,
            batch_size=20,
            callbacks=callbacks,
            verbose=0
        )
        y_pred_soft = model.predict(x_train)
        y_predict = np.array(np.argmax(y_pred_soft, axis=1))
        y_train_vec = np.array(np.argmax(y_train, axis=1))
        compare = (y_predict == y_train_vec)
        print("Train labels", y_train_vec, "Train Predict", y_predict)
        print("Our calculated train accuracy", np.sum(compare) / len(y_train_vec))

        s = model.evaluate(x_test, y_test, verbose=1)
        y_test_soft = model.predict(x_test)
        y_test_predict = np.array(np.argmax(y_test_soft, axis=1))
        print("Test labels", np.array(np.argmax(y_test, axis=1)), "Test Predict", y_test_predict,
              "Our calculated test accuracy",
              np.sum(y_test_predict == np.array(np.argmax(y_test, axis=1))) / len(y_test_predict))
        if (s[1] > bestAccuracy):
            bestAccuracy = s[1]
        accuracy = accuracy + s[1]
    print("Best Accuracy = ", bestAccuracy)
    print("Average Accuracy = ", accuracy / iterations)

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
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1,activation="sigmoid")(x)
    return x + res

def build_model(
        input_shape,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_units,
        dropout=0.4,
        mlp_dropout=0.4,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    return keras.Model(inputs, outputs)

class PredictionCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    y_pred = self.model.predict(self.validation_data[0])
    print('prediction: {} at epoch: {}'.format(y_pred, epoch))

