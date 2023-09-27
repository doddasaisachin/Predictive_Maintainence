import pandas as pd
import warnings 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('./Maintenance.csv')

df.head()

len(df['Product ID'].unique())

len(df['UDI'].unique())

new_df=df.copy()

new_df.head()

new_df.drop(['UDI','Product ID'],axis=1,inplace=True)

new_df.head()

new_df['Type'].value_counts()



enc=LabelEncoder()

enc_type=enc.fit_transform(df['Type'])

np.unique(enc_type)

new_df['Type']=enc_type

new_df.head()

new_df.dtypes

new_df.describe()

new_df.head(2)

from sklearn.model_selection import train_test_split

X=new_df.iloc[:,:6]

X.head(1)

Y=new_df['Machine failure']





smote_over_sampling=SMOTE()

over_sampled_data=smote_over_sampling.fit_resample(X,Y)

new_X=over_sampled_data[0]

new_X.head(2)

new_Y=over_sampled_data[1]
new_Y

new_X.shape

new_Y.shape

new_Y.value_counts()



xtrain,xtest,ytrain,ytest=train_test_split(new_X,new_Y,test_size=0.2)

xtrain.shape,xtest.shape,ytrain.shape,ytest.shape



forest_model=RandomForestClassifier()

forest_model.fit(xtrain,ytrain)


ypred=forest_model.predict(xtest)

accuracy_score(ytest,ypred)*100

new_df.head(2)

failure_df=new_df.iloc[:,-5:]

failure_df.head(2)

new_df[failure_df['TWF']==1]

new_df[failure_df['TWF']==1]['Machine failure'].value_counts()

new_df[failure_df['HDF']==1]['Machine failure'].value_counts()

new_df[failure_df['PWF']==1]['Machine failure'].value_counts()

new_df[failure_df['OSF']==1]['Machine failure'].value_counts()

new_df[failure_df['RNF']==1]['Machine failure'].value_counts()

new_df[failure_df['RNF']==1]

forest_model.predict(new_df[failure_df['RNF']==1].iloc[:,:6])

new_df.drop('RNF',axis=1,inplace=True) 

new_df.head(3)

df.head(2)


warnings.filterwarnings('ignore')

def predict_machine_failure(model,Type,Air_temp,Process_temp,Rotational_speed,Torque,Tool_wear,TWF,HDF,PWF,OSF):
    if (TWF==1) or (HDF==1) or (PWF==1) or (OSF==1):
        return True
    else:
        predicted=model.predict([[Type,Air_temp,Process_temp,Rotational_speed,Torque,Tool_wear]])
    return True if predicted[0]==1 else False

# a,b,c,d,e,f,g,h,i,j,k=new_df.iloc[1,:].values

# predict_machine_failure(forest_model,a,b,c,d,e,f,h,i,j,k)

import pickle

pickle.dump(forest_model,open('random_forest_model.pkl','wb'))

# temp=pickle.load(open('./random_forest_model.pkl','rb'))

# predict_machine_failure(temp,a,b,c,d,e,f,h,i,j,k)

