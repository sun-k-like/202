import tensorflow as tf
import pandas as pd

#Data Loading
irisfile = 'iris.csv'
iris = pd.read_csv(irisfile)
iris.head()

#graophical representation
import seaborn as sns
import matplotlib.pyplot as plt 
sns.pairplot(iris, hue='class')
plt.show()

#columns
iris = pd.get_dummies(iris)
iris.head()

#Variables
indep = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
depen = iris[['class_Iris-setosa', 'class_Iris-versicolor', 'class_Iris-virginica']]
print(indep.shape, depen.shape) 

#Model Creation
X=tf.keras.layers.Input(shape=[4])
Y=tf.keras.layers.Dense(3, activation='softmax')(X)
model=tf.keras.models.Model(X,Y)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

#Model Training
model.fit(indep, depen, epochs=1000, verbose=0)
model.fit(indep, depen, epochs=10)

#Prediction
print("Predict:", model.predict(indep[45:55]))
print("Labels:", depen[45:55])