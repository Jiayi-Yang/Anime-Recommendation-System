import pandas as pd
import numpy as np

originalData = pd.read_csv('C:/Users/kexic/Documents/AI/anime-recommendations-database/anime_new.csv')
genre = set()

#get the set
genreVal = originalData['genre'].values
for row in genreVal:
    if isinstance(row,str):
        rowList = row.split(',')
        for ele in rowList:
            genre.add(ele.strip())

rows = len(originalData)

for types in list(genre):
    originalData[types] = np.zeros(rows)

for i in xrange(rows):
    tmp = originalData.ix[i]['genre']
    if isinstance(tmp,str):
        for types in tmp.split(','):
            originalData.ix[i][types.strip()] = 1
