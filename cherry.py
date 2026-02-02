import tensorflow as tf
import pandas as pd
import numpy as np
datafile = "cherry.csv"
cherry = pd.read_csv(datafile)
cherry.head()
indep = cherry[['day']]
depen = cherry[['product']]
print(indep.shape, depen.shape)
X=tf.keras.layers.Input(shape=[1])
Y=tf.keras.layers.Dense(1)(X)
model=tf.keras.models.Model(X,Y)
model.compile(loss='mse')
model.fit(indep, depen, epochs=20000, verbose=0)
model.fit(indep, depen, epochs=20)

#predict( )
print(model.predict(indep))
x_new = np.array([[16],[51]])
print(model.predict(x_new))

