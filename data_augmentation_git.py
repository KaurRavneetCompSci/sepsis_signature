import numpy as np
import pandas as pd
import similaritymeasures
from tsaug import TimeWarp
from tsaug import Crop, Quantize, Drift, Reverse
df = pd.DataFrame({"timestamp": [1, 2],"cas_pre": [10, 20],"fl_rat": [30, 40]})
#my_aug = ( Drift(max_drift=(0.1, 0.5)))
preparedData = pd.read_csv("/Users/harpreetsingh/Harpreet/Ravneet/Sepsis_Signature/generalised-signature-method/signature.csv",low_memory=False)


my_aug = (TimeWarp()*2)
dfNumpy = df[["timestamp","cas_pre","fl_rat"]].to_numpy()
aug = my_aug.augment(dfNumpy)
print("Input:")
print(df[["timestamp","cas_pre","fl_rat"]].to_numpy()) #debug
print("Output:")
print(aug)