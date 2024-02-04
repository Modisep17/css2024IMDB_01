# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#MY FIRST CODING ASSIGNMENT,I hope it few hours this will be done, its 7am,sat
print ("hello Phenyo")
print[2]     ]
print (2)
#Getting started
import pandas as pd
df=pd.read_csv("movie_dataset.csv")
print (df)
print(df.info())
print(df.describe())
# Allows myself to see all rows
pd.set_option('display.max_rows',None)
print(df)
df.drop(['Description'],inplace=True,axis=1)
print(df)
df.drop(['Votes'],inplace=True,axis=1)
df.drop(['Metascore'],inplace=True,axis=1)
print(df)
df.drop(['Runtime (Minutes)'],inplace=True,axis=1)
print(df)
print(df.info())
print(df.describe())
# Allows me to see all rows
pd.set_option('display.max_rows',None)
print(df)# Filtering data
print(df[df['Rating'] >8.5])
# Filter data for Director (Director == 'Christopher Nolan')
Christopher_Nolan = df[df['Director'] == 'Christopher Nolan']
print(df[df['Rating'] >=8])
Christopher_Nolan = df[df['Director'] == 'Christopher Nolan']
print(df[df['Director'] == 'Christopher Nolan'])
df=df[df['Director'] == 'Christopher Nolan']
x = df["Rating"].median()
print(x)
df=pd.read_csv("movie_dataset.csv")
print(df)
df.drop(['Description'],inplace=True,axis=1)
print(df)
df.drop(['Votes'],inplace=True,axis=1)
df.drop(['Metascore'],inplace=True,axis=1)
print(df)
df.drop(['Runtime (Minutes)'],inplace=True,axis=1)
print(df)
print(df.info())
print(df.describe())
# Allows me to see all rows
pd.set_option('display.max_rows',None)
print(df)
x = df['Revenue (Millions)'].mean()
df['Revenue (Millions)'].fillna(x, inplace = True) 
print(df)
print(df[df['Rating'] >=9])
# Filtering my data
print(df[df['Year'] == 2006])
print(df[df['Year'] == 2016])
Y2006=44
x2016=297
X=44
Y=297
subtract=Y-X
print(subtract)
print(subtract/X) 
Percentage= (subtract/X)*100
print (Percentage)
print(df)
pd.set_option('display.max_rows',None)
print(df)
print(df.info)
print(df.describe)
x = df['Revenue (Millions)'].mean()
print (x)
df['Revenue (Millions)'] = df[C].apply(lambda x: pd.to_numeric(x, errors = 'ignore'))
x = df['Revenue (Millions)'].mean()
print (x)
pd.set_option('display.max_rows',None)
print(df)
print(sum('Revenue (Millions)'))
df = df.astype({'Revenue (Millions)':'float'})
df['Revenue (Millions)'] = df['Revenue (Millions)'].astype(float)
print(sum('Revenue (Millions)'))
print(df.info())
df["Revenue (Millions)"] = pd.to_numeric(df["Revenue (Millions)"])
print(df.info())
print(sum('Revenue (Millions)'))
#Attempt again, the genre 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df1 = df.filter(['Genre'], axis=1)
print(df1)
df2 = df1['Genre'].str.get_dummies(',')
print(df2.describe())
pd.set_option('display.min_row',None)
print(df2)
print(df2.info())
print(df2.describe())
pd.unique(df["Genre"].str.split(",", expand=True).stack())
df["Genre"].str.split(",", expand=True)
#Attempt find the common actor in all movies
df2 = df.filter(['Title','Actors'], axis=1)
from collections import Counter
df2 = df.filter(['Title','Actors'], axis=1)
df2["Actors"].str.split(",", expand=True)
col = df2["Actors"]
cnt = Counter(df2["Actor"] for df2 in list)
print(cnt)
list=pd.unique(df2["Actors"].str.split(" ,", expand=True).stack())
print(list)
#Trying internet suggestions, name of actor found is Matthew, with appearances in 26 movies
cnt = Counter(df2["Actor"] for df2 in list)
print(cnt)
c=Counter(list)
print(c.most_common(1)[0][0])
max_occurence=c.most_common(1)[0][1]
print(c.most_common(1)[0][0])
#Trying another internet metho for common actor in all movies
 import numpy as np
 import pandas as pd
from collections import Counter
df4 = df.filter(['Actors'], axis=1)
df4["Actors"].str.split(",", expand=True)
df5=df4["Actors"].str.split(",", expand=True)
cnt = Counter(df4["Actor"] for df4 in list)
print(df2)
df6=df2["Actors"].str.split(",", expand=True)
df6=pd.unique(df2["Actors"].str.split(",", expand=True).stack())
l=pd.unique(df2["Actors"].str.split(",", expand=True).stack())
print(df6)
data=np.array(list)
print(data)
import collections, numpy
counter = collections.Counter(data)
>>> counter
#correlation attempts as a first-timer in programming. It is not easy, Its an attempt, not sure method
from numpy import mean
from numpy import std
from numpy.random import randn
from numpy.random import seed
from matplotlib import pyplot
seed(1)
data1=(df['Year'])
data2=(df['Rating'])
pyplot.scatter(data1, data2)
data3=(df['Title'])
pyplot.scatter(data1, data3)
data4=(df['Revenue (Millions)'])
pyplot.scatter(data1, data4)
pyplot.scatter(data2, data3)
print('data4: mean=%.3f stdv=%.3f' % (mean(data4), std(data4)))
print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
pyplot.scatter(data2, data4),plt.xlabel("Rating"),plt.ylabel("Revenue (Millions)"), 
plt.ylabel("Rating")
plt.xlabel("Revenue(Millions)")
plt.ylabel("Rating")
from scipy.stats import pearsonr
corr, _ = pearsonr(data2, data4)
print('Pearsons correlation: %.3f' % corr)
#Coding and sleep are like oil and water, the minute you start the is no end,its 2am sunday.

