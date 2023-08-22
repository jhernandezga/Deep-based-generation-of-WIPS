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

file = 'wips_reference.csv'

frame = pd.read_csv(file)


#print(frame.iloc[483])

frame['species name'] = frame['species name'].str.lower()

frame.sort_values(by=['category','photo'], inplace= True)
file_name = 'MarksData2.xlsx'

cat = frame['category'].value_counts()
species = frame['species name'].value_counts()

counts_frame = pd.concat([cat, species], axis=1)

counts_frame.to_excel(file_name)

category = 13

print(frame.tail())
frame = frame[frame['category'] == category]



if len(frame) == 0:
    raise ValueError('Category not found: Category [1-144]')


#print(frame.iloc[483])
#print(filter.tail())
print(len(frame))