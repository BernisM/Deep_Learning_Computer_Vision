import numpy as np 
from numpy import genfromtxt

fld = r'C:\Users\massw\OneDrive\Bureau\Programmation\Python_R\Computer-Vision-with-Python\DATA'
file = r"bank_note_data.txt"
path = '{}\{}'.format(fld,file)

data = genfromtxt(path,delimiter=',')

data

labels = data[:,4]

features = data[:,0:4]

x = features
y = labels

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

from sklearn.preprocessing import MinMaxScaler

scaler_object = MinMaxScaler()

scaler_object.fit(x_train)

scaled_x_train = scaler_object.transform(x_train)
scaled_x_test = scaler_object.transform(x_test)

scaled_x_test.min()

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(4,input_dim=4,activation='relu'))
model.add(Dense())


















