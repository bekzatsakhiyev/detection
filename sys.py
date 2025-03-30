import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(42)
n_samples = 1000
data = {
    'camera_access': np.random.randint(0, 2, n_samples),
    'sms_access': np.random.randint(0, 2, n_samples),
    'internet_access': np.random.randint(0, 2, n_samples),
    'location_access': np.random.randint(0, 2, n_samples),
    'contacts_access': np.random.randint(0, 2, n_samples),
}
labels = ((data['camera_access'] + data['sms_access'] + data['internet_access'] +
          data['location_access'] + data['contacts_access']) > 3).astype(int)
df = pd.DataFrame(data)
df['label'] = labels
print("Распределение классов в датасете:")
print(df['label'].value_counts())

X = df.drop('label', axis=1).values
y = df['label'].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=16,
    validation_split=0.2,
    verbose=1,
    callbacks=[early_stopping]
)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Точность на тестовых данных: {test_accuracy:.4f}")
plt.plot(history.history['accuracy'], label='Точность на обучении')
plt.plot(history.history['val_accuracy'], label='Точность на валидации')
plt.title('Точность модели')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.show()
plt.plot(history.history['loss'], label='Потери на обучении')
plt.plot(history.history['val_loss'], label='Потери на валидации')
plt.title('Потери модели')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()
plt.show()
predictions = model.predict(X_test, batch_size=16)
predictions = (predictions > 0.5).astype(int)
print("Первые 5 примеров:")
for i in range(5):
    print(f"Реальный класс: {y_test[i]}, Предсказанный класс: {predictions[i][0]}")
print("\nПримеры с классом 1:")
indices = np.where(y_test == 1)[0][:5]
for i in indices:
    print(f"Реальный класс: {y_test[i]}, Предсказанный класс: {predictions[i][0]}")