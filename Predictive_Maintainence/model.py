import pandas as pd
import warnings 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('./Maintenance.csv')

new_df=df.copy()
new_df.drop(['UDI','Product ID'],axis=1,inplace=True)

enc=LabelEncoder()
enc_type=enc.fit_transform(df['Type'])
new_df['Type']=enc_type

from sklearn.model_selection import train_test_split
X=new_df.iloc[:,:6]
Y=new_df['Machine failure']

smote_over_sampling=SMOTE()
over_sampled_data=smote_over_sampling.fit_resample(X,Y)
new_X=over_sampled_data[0]
new_Y=over_sampled_data[1]

xtrain,xtest,ytrain,ytest=train_test_split(new_X,new_Y,test_size=0.2)

forest_model=RandomForestClassifier()
forest_model.fit(xtrain,ytrain)

new_df.drop('RNF',axis=1,inplace=True) 

warnings.filterwarnings('ignore')

def predict_machine_failure(model,Type,Air_temp,Process_temp,Rotational_speed,Torque,Tool_wear,TWF,HDF,PWF,OSF):
    if (TWF==1) or (HDF==1) or (PWF==1) or (OSF==1):
        return True
    else:
        predicted=model.predict([[Type,Air_temp,Process_temp,Rotational_speed,Torque,Tool_wear]])
    return True if predicted[0]==1 else False

import pickle
pickle.dump(forest_model,open('random_forest_model.pkl','wb'))