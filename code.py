import tensorflow as tf
import pandas as pd
from keras import Sequential
from keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error


columns = ['Per capita crime rate by town', 'Land Zoned', 'Proportion of residential land zoned', 'Charles River', 'Nitric oxides concentration ', 'Average Rooms', 'AGE (in years)', 'Distance to Employeement Center', 'Radial Highway', 'Tax', 'Pupil-teacher ratio', 'Blacks propotion', 'Lower status of the population', 'Median value of owner-occupied']
df = pd.read_csv('/kaggle/input/housing/housing.csv', header=None, delimiter=r"\s+", names=columns)
df.head()

df.describe()

df.info()

cat_col = ['Charles River','Radial Highway']
num_col = ['Per capita crime rate by town','Land Zoned','Proportion of residential land zoned','Nitric oxides concentration ','Average Rooms','AGE (in years)','Distance to Employeement Center','Tax','Pupil-teacher ratio','Blacks propotion','Lower status of the population']
tot_cols = cat_col + num_col
plt.subplots(figsize=(13, 10))
sns.heatmap(df.corr(),annot=True)

y = df['Median value of owner-occupied']
df = df.drop("Median value of owner-occupied", axis=1)

Pricing = StandardScaler()
scaled_cols = Pricing.fit_transform(df[num_col])

scaled_cols = pd.DataFrame(scaled_cols)
scaled_cols.columns = num_col
scaled_cols.head()

df2 = df[cat_col].join(scaled_cols)
df2.head()

from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(df2,y,test_size=0.2)

model = Sequential()
model.add(Dense(13, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

epochs=int(input("ENTER NUMBER OF PREDICTIONS"))
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001), loss='mse')
model.fit(X_train, y_train, epochs)


calculate = model.predict(X_test)

result = mean_squared_error(y_test,calculate)

print(result)
