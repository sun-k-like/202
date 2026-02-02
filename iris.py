import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# TensorFlow 경고 메시지 끄기
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 1. 데이터 로드 (컬럼명이 없는 파일이므로 직접 이름을 지정해줍니다)
column_names = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Class']
iris = pd.read_csv('iris.csv', names=column_names)

print("성공적으로 설정된 컬럼명:", iris.columns.tolist())

# 2. 그래프 출력
sns.pairplot(iris, hue='Class')
plt.show()

# 3. 원-핫 인코딩 (Class 컬럼을 0과 1로 변환)
iris = pd.get_dummies(iris)

# 4. 독립변수(X)와 종속변수(Y) 분리
# pd.get_dummies 결과로 생성된 실제 컬럼명을 확인하며 넣어야 합니다.
indep = iris[['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']]
depen = iris[['Class_Iris-setosa', 'Class_Iris-versicolor', 'Class_Iris-virginica']]

print("독립변수 형태:", indep.shape, "종속변수 형태:", depen.shape)

# 5. 모델 구조 생성
X = tf.keras.layers.Input(shape=[4])
Y = tf.keras.layers.Dense(3, activation='softmax')(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# 6. 모델 학습
print("\n--- 학습 시작 ---")
model.fit(indep, depen, epochs=1000, verbose=0)
model.fit(indep, depen, epochs=10)

# 7. 예측 및 결과 확인
print("\n--- 예측 결과 (45~55번 데이터) ---")
print("Predict:\n", model.predict(indep[45:55]))
print("Labels:\n", depen[45:55])