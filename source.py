import csv
import pandas as pd
import numpy as np
import os
from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
import matplotlib.pyplot as plt
plt.rc("font", size=14)
linReg = linear_model.LinearRegression()
logReg = linear_model.LogisticRegression()


def cleanData(v):


    #modify M to 0
    cond = v['Sex'] == 'M'
    v.loc[cond, 'Sex'] = 0
    #modify F to 1
    cond = v['Sex'] == 'F'
    v.loc[cond, 'Sex'] = 1
    
    #modify Flat to 0
    cond = v['ST_Slope'] == 'Flat'
    v.loc[cond, 'ST_Slope'] = 0

    #modify Up to 1
    cond = v['ST_Slope'] == 'Up'
    v.loc[cond, 'ST_Slope'] = 1

    #modify Flat to -1
    cond = v['ST_Slope'] == 'Down'
    v.loc[cond, 'ST_Slope'] = -1




    #modify TA to 0
    cond = v['ChestPainType'] == 'TA'
    v.loc[cond, 'ChestPainType'] = 0

    #modify ATA to 1
    cond = v['ChestPainType'] == 'ATA'
    v.loc[cond, 'ChestPainType'] = 1

    #modify NAP to 2
    cond = v['ChestPainType'] == 'NAP'
    v.loc[cond, 'ChestPainType'] = 2

    #modify ASY to 3
    cond = v['ChestPainType'] == 'ASY'
    v.loc[cond, 'ChestPainType'] = 3


    #modify Y to 1 within ExerciseAngina Column
    cond = v['ExerciseAngina'] == 'Y'
    v.loc[cond, 'ExerciseAngina'] = 1

    #modify N to 1 within ExerciseAngina Column
    cond= v['ExerciseAngina'] == 'N'
    v.loc[cond, 'ExerciseAngina'] = 0

    #modify Normal to 0 within RestingECG Column
    cond = v['RestingECG'] == 'Normal'
    v.loc[cond, 'RestingECG'] = 0

    #modify ST to 1 within RestingECG Column
    cond = v['RestingECG'] == 'ST'
    v.loc[cond, 'RestingECG'] = 1

    #modify LVH to 2 within RestingECG Column
    cond = v['RestingECG'] == 'LVH'
    v.loc[cond, 'RestingECG'] = 2



    return v

#read data from csv file
main_df = pd.read_csv('data/cleanHeart.csv')

#main_df = cleanData(main_df)

#os.makedirs('data', exist_ok=True)
#main_df.to_csv('data/cleanHeart.csv', index=False)


regu = linReg.fit(main_df[['Age','Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']],main_df['HeartDisease'])

tester = regu.predict([[65,0,3,140,306,1,0,87,1,1.5,0]])


print(tester)
