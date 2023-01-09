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
# scaled_x_test --> New data que le modele n'a jamais vu
scaled_x_test = scaler_object.transform(x_test)

scaled_x_test.min()

from keras.models import Sequential
from keras.layers import Dense

# Creates model
model = Sequential()
# 8 Neurons, expects input of 4 features. 
# 4 neurones, 4 dimensions, activation = ReLU
model.add(Dense(4,input_dim=4,activation='relu'))
# Add another Densely Connected layer (every neuron connected to every neuron in the next layer)
model.add(Dense(8,activation='relu'))
# Last layer simple sigmoid function to output 0 or 1 (our label)
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# Train/Fit the model
# 1 epochs --> Parcourir données d'entraitements une fois
model.fit(scaled_x_train,y_train,epochs=50,verbose=2)

model.metrics_names
from sklearn.metrics import confusion_matrix,classification_report

predictions = np.argmax(model.predict(scaled_x_test),axis=-1)

confusion_matrix(y_test,predictions)

print(classification_report(y_test,predictions))

## Saving and Loading Models

model.save('mymediumodel.h5')

from keras.models import load_model

new_model = load_model('mymediumodel.h5')

new_model.predict(scaled_x_test)















