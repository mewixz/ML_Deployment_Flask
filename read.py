
# Generate a scatter plot and returns the figure 

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_plot(): 
  
    data = { 
        'a': np.arange(50), 
        'c': np.random.randint(0, 50, 50), 
        'd': np.random.randn(50) 
    } 
  
    data['b'] = data['a'] + 10 * np.random.randn(50) 
    data['d'] = np.abs(data['d']) * 100
  
    plt.scatter('a', 'b', c='c', s='d', data=data) 
    plt.xlabel('X label') 
    plt.ylabel('Y label') 
  
    return plt 






import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error




data = pd.read_csv('https://raw.githubusercontent.com/cambridgecoding/machinelearningregression/master/data/bikes.csv')
print(data.head(3))

data['date'] = data['date'].apply(pd.to_datetime)
data['year'] = [i.year for i in data['date']]
data['month'] = [i.month_name()[0:3] for i in data['date']]
data['day'] = [i.day_name()[0:3] for i in data['date']]

import matplotlib.pyplot as plt
import seaborn as sns


figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
bp1 = sns.barplot(data=data, x='day', y='count', hue='year', ax=ax1)
bp2 = sns.barplot(data=data, x='month', y='count', hue='year', ax=ax2)

plt.savefig("./static/images/bar0_plot.png")


figure = plt.figure(figsize=(8,6))

pp = sns.pairplot(data=data, y_vars=['count'], x_vars=['temperature', 'humidity', 'windspeed'], height=4)

plt.savefig("./static/images/bar_plot.png")


#plt.show()
#/home/alexander/anaconda3/envs/bell/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
#  return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval

print('*'*50)
x = data[['temperature', 'humidity', 'windspeed']]
y = data['count']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

print(X_train.shape, y_train.shape)


classifier = xgb.XGBRegressor(random_state=42)


classifier.fit(X_train, y_train)
print('%'*100)


predictions = classifier.predict(X_test)
print(f'R^2 score: {r2_score(y_true=y_test, y_pred=predictions):.2f}')
print(f'MAE score: {mean_absolute_error(y_true=y_test, y_pred=predictions):.2f}')
print(f'EVS score: {explained_variance_score(y_true=y_test, y_pred=predictions):.2f}')

figure = plt.figure(figsize=(8,6))

rp = sns.regplot(x=y_test, y=predictions, ci=99, marker="o", line_kws=dict(color="r"))
plt.savefig("./static/images/reg_plot.png")


import pickle
with open('./bike_model_xgboost.pkl', 'wb') as file:
    pickle.dump(classifier, file)

print('done')