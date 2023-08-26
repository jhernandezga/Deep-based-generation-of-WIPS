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

file = 'wips_referenceXX.csv'

frame = pd.read_csv(file)


#print(frame.iloc[483])

frame['species name'] = frame['species name'].str.lower()

frame.sort_values(by=['category','photo'], inplace= True)
file_name = 'MarksData4.xlsx'

cat = frame['category'].value_counts()
species = frame['species name'].value_counts()

#counts_frame = pd.concat([cat, species], axis=1)
counts_frame = frame['category'].value_counts()

category_genre_mapping = frame.drop_duplicates(subset=['category']).set_index('category')['Genre'].to_dict()
category_species_mapping = frame.drop_duplicates(subset=['category']).set_index('category')['Espèce'].to_dict()
category_subspecies_mapping = frame.drop_duplicates(subset=['category']).set_index('category')['sous espèce'].to_dict()

category_table = pd.DataFrame({'Category Name': counts_frame.index,
                               'Genus': counts_frame.index.map(category_genre_mapping),
                               'Species': counts_frame.index.map(category_species_mapping),
                               'Subspecies': counts_frame.index.map(category_subspecies_mapping),
                               'Count': counts_frame.values})

category_table.to_excel(file_name)

category = 13

print(frame.tail())
frame = frame[frame['category'] == category]



if len(frame) == 0:
    raise ValueError('Category not found: Category [1-144]')


#print(frame.iloc[483])
#print(filter.tail())
print(len(frame))