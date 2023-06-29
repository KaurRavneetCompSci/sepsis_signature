import linecache
from definitions import *
from collections import defaultdict
from tsaug import TimeWarp
import pandas as pd

enablingRandomize = False
print('---Data Preparation Called---')
class DataPreparation:
    def __init__(self, filePath):
        self.filePath = filePath
    def PrintException(self):
        import sys
        exc_type, exc_obj, tb = sys.exc_info()
        f = tb.tb_frame
        lineno = tb.tb_lineno
        filename = f.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))
    def read_file_data(filename, type_):
        with open(filename) as acc_file:
            _acc_file_data = acc_file.read().strip().split('\n')
            acc_file_data = []
            for line in _acc_file_data:
                acc_file_data.append(tuple(type_(i) for i in line.strip().split(' ')))
        return acc_file_data
    def read_prepare_data_auto(self,patientCaseUHID,caseType,conditionCase,folderName):
        try:
            #path = os.getcwd()
            path = self.filePath
            seperator = '_'
            filePath = path+folderName+str(patientCaseUHID)+"/"
            fileName = filePath+caseType+seperator+str(patientCaseUHID)+seperator+'intermediate_checkpoint_new_5.csv'
            #print('fileName=>'+fileName)
            preparedData = pd.read_csv(fileName,low_memory=False)
            return fileName, preparedData
        except IOError as e:
            print('No data for this uhid in data preparation', e)
        except Exception as e:
            print('Exception in read_prepare_data', e)
            self.PrintException()

    #This method augments the timeseries data using warping
    def augmentData(self,dict_Signature,labels_list):
        df = pd.DataFrame({"timestamp": [1, 2], "cas_pre": [10, 20], "fl_rat": [30, 40]})
        my_aug = (TimeWarp() * 2)
        dfNumpy = df[["timestamp", "cas_pre", "fl_rat"]].to_numpy()
        aug = my_aug.augment(dfNumpy)
        print("Input:")
        print(df[["timestamp", "cas_pre", "fl_rat"]].to_numpy())  # debug
        print("Output:")
        print(aug)
        return aug

    #This method reads two directories containing sepsis and nosepsis patients.  Reads CSV from both folders that contain
    #one folder for a given patient id - UHID. It then creates a CSV file containing physiological data
    def prepareData(self, filename,dataRead):
        finalSetDS = pd.read_csv(self.filePath+filename,low_memory=False)
        # print('Length of balanced dataset', len(finalSetDS))
        preparedData = pd.DataFrame()
        labels_list = []
        i = 1
        typeOfCase = ""
        sepsisCase = 1
        noSepsisCase = 0
        dict_Signature = defaultdict(list)
        for row in finalSetDS.itertuples():
            try:
                #print(i,'---->',getattr(row, 'uhid'), getattr(row, 'dischargestatus'))
                i=i+1
                patientCaseUHID = getattr(row, 'uhid')
                #print('patientCaseUHID', patientCaseUHID)
                caseType = getattr(row, 'dischargestatus')
                #print('caseType',caseType,type(caseType))
                if caseType == sepsisCase:
                    folderName = "/Sepsis_Cases/"
                    conditionCase = "(dischargestatus = 'Death' or dischargestatus = 'LAMA' or dischargestatus = 'Sepsis' )"
                    typeOfCase = "Death"
                elif caseType == noSepsisCase:
                    folderName = "/NoSepsis_Cases/"
                    conditionCase = "(dischargestatus = 'Discharge')"
                    typeOfCase = "Discharge"
                #print('patientCaseUHID',patientCaseUHID,'caseType',typeOfCase,'conditionCase',conditionCase,'folderName',folderName)
                # uncomment below to generate data first time
                #fileName,uhidDataSet = prepare_data(con,patientCaseUHID,typeOfCase,conditionCase,folderName)
                # uncomment below in case csv data is already generated and now lstm needs to be executed
                patientCaseUHID = str(patientCaseUHID)
                fileName,preparedData = self.read_prepare_data_auto(patientCaseUHID,typeOfCase,conditionCase,folderName)
                if caseType == sepsisCase:
                    preparedData['sepsis'].replace(to_replace = 0.0,  method='bfill', inplace=True, limit = dataRead)
                    sepsis = preparedData[preparedData['sepsis'] == 1]
                    sepsis = sepsis[['uhid','heartrate','spo2']]
                    sepsis = sepsis.fillna(method='ffill')
                    sepsis = sepsis.astype('float')
                    labels_list.append(1.0)
                    data_list = []
                    #print(len(sepsis))
                    uhidKey = []
                    counterData =0
                    for index, row in sepsis.iterrows():
                        uhidKey = row['uhid']
                        tuple = (row['heartrate'], row['spo2'],counterData)
                        counterData= counterData+1
                        data_list.append(tuple)
                    uhidCompositeKey = str(uhidKey)+ '_sepsis'
                    dict_Signature[uhidCompositeKey].append(data_list)
                else:
                    preparedData = preparedData[preparedData['heartrate'] >= 0]
                    preparedData = preparedData[preparedData['spo2'] >= 0]
                    preparedData = preparedData.head(dataRead+1)
                    sepsis = preparedData[['uhid','heartrate','spo2']]
                    sepsis = sepsis.astype('float')
                    labels_list.append(0.0)
                    data_list = []
                    #print(len(sepsis))
                    uhidKey = []
                    counterData =0
                    for index, row in sepsis.iterrows():
                        uhidKey = row['uhid']
                        tuple = (row['heartrate'], row['spo2'],counterData)
                        counterData= counterData+1
                        data_list.append(tuple)
                    uhidCompositeKey = str(uhidKey) + '_nosepsis'
                    dict_Signature[uhidCompositeKey].append(data_list)
            except Exception as e:
                print('Exception in prediction_data_death_discharge', e)
                self.PrintException()
        print(len(dict_Signature))
        #print(len(labels_list))
        X = (dict_Signature)
        Y = (labels_list)
        return X,Y


