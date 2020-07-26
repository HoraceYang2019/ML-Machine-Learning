# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 11:49:09 2018

@author: USER
"""

import matplotlib.pyplot as plt
year = [1950, 1970, 1990, 2010]
population = [2.519, 3.692, 5.263, 6.972]
plt.plot(year, population)
plt.show()
plt.scatter(year, population)
plt.show()

help(plt.hist)
values = [0, 0.6, 1.4, 1.6, 2.2, 2.5, 2.6, 3.2, 3.5, 3.9, 4.2, 6]
plt.hist(values, bins = 3)
plt.show()

plt.plot(year, population)
plt.show()

plt.plot(year, population)
plt.xlabel('Year')
plt.ylabel('Population')
plt.show()


plt.plot(year, population)
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('World Population Projections')
plt.show()


#
plt.plot(year, population)
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('World Population Projections')
plt.yticks([0,2,4,6,8,10])
plt.show()


# In[161]:


plt.plot(year, population)
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('World Population Projections')
plt.yticks([0,2,4,6,8,10],
           ['0','2B','4B','6B','8B','10B'])
plt.show()


# <font color="red">Add historical data</font>

# In[162]:

population = [1.0,1.262,1.650] + population
year = [1800,1850,1900] + year
plt.plot(year, population)
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('World Population Projections')
plt.yticks([0,2,4,6,8,10],
           ['0','2B','4B','6B','8B','10B'])
plt.show()

# In[163]:

plt.fill_between(year,population,0,color='green')
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('World Population Projections')
plt.yticks([0,2,4,6,8,10],
           ['0','2B','4B','6B','8B','10B'])
plt.show()