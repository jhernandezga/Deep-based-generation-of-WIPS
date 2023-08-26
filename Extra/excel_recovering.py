from __future__ import print_function, division
import os
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


frame = pd.read_excel("IdentifiantValPhB15022017.xlsx")

#print(frame.head(5))
a = frame.drop(['Unnamed: 28','Unnamed: 9','GeoRef'], axis = 1)
#print(a.info())
a['Date photo'] = a['Date photo'].astype('str')
#a['Date photo'] = pd.to_datetime(a['Date photo'])
a["M/f/ND(0/1/2)"] = a["M/f/ND(0/1/2)"].astype('string')
a[['Infra ordre','Super famille','sous genre','Code pays','Lieu de capture','Det']] = a[['Infra ordre','Super famille','sous genre','Code pays','Lieu de capture','Det']].fillna("ND")
for x in a.index:
  if a.loc[x, 'Date photo'] == '02/12/215':
    a.loc[x, 'Date photo'] =  '02/12/2015'
  elif a.loc[x, 'Date photo'] == '10/12/215':
    a.loc[x, 'Date photo'] = '10/12/2015'
  if a.loc[x, 'Date photo'] == 'ND':
    a.loc[x, 'Date photo'] = np.nan
  if  a.loc[x, "M/f/ND(0/1/2)"] == 'ND':
    a.loc[x, "M/f/ND(0/1/2)"] = '2'
    
#a['sous espèce'] = a.fillna('ND')      
a['individus'] = a['individus'].astype('string') 
a['Catégorie'] = a['Catégorie'].astype('int16') 
a["M/f/ND(0/1/2)"] = a["M/f/ND(0/1/2)"].astype('int8')
#a['Date photo'] = pd.to_datetime(a['Date photo'],infer_datetime_format=True)
a = a.drop_duplicates()    
#print(a.head())
#print(a.describe())
#print(a.info())
#print(a.loc[1867])

pd.set_option('display.max_rows', 200)
print(a['Espèce'].value_counts())
print(a['Catégorie'].value_counts())
b = a[['Espèce','Catégorie']]
#print(b.corr())


print(a.info())

#histogram = a['Espèce'].value_counts().hist()
#fig1 = histogram.get_figure()
#fig1.savefig('species.jpg')

data_reference = a[['photo N°','Genre','sous genre','Espèce','sous espèce',"Nom commun de l'espèce","Catégorie",'M/f/ND(0/1/2)']]
data_reference["Nom commun de l'espèce"] = data_reference["Nom commun de l'espèce"].astype('category')
data_reference["Catégorie"] = data_reference["Catégorie"].astype('category')
data_reference["photo N°"] = data_reference["photo N°"].astype('int16')

print(data_reference.info())
print(data_reference.head())

print(data_reference["Catégorie"].value_counts())
print(data_reference["Nom commun de l'espèce"].value_counts())
#print(a.head())
data_reference.rename(columns = {"Nom commun de l'espèce":'species name','Catégorie':'category','photo N°':'photo'}, inplace = True)
data_reference.to_csv('wips_referenceXX.csv',index = False)