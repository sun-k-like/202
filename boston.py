import tensorflow as tf
import pandas as pd

#Data Loading
datafile = "boston.csv" 
boston = pd.read_csv(datafile)
print(boston.columns)
boston.head()

#Variables
indep = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']]
depen = boston[['medv']]
print(indep.shape, depen.shape)

#Model Creation
X=tf.keras.layers.Input(shape=[13])
Y=tf.keras.layers.Dense(1)(X)
model=tf.keras.models.Model(X,Y)
model.compile(loss='mse')  

#Model Training
model.fit(indep, depen, epochs=1000, verbose=0)
model.fit(indep, depen, epochs=20)

#Prediction
print(model.predict(indep[0:5]))
print(depen[0:5])

#weights
print(model.get_weights())