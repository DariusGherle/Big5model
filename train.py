import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import joblib

# Încarcă datele
df = pd.read_csv('data-final.csv', sep='\t')

# Listează toți cei 5 factori
factors = ['EXT', 'EST', 'AGR', 'CSN', 'OPN']

# Creează foldere pentru rezultate dacă nu există
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("scalers", exist_ok=True)

# salv matriele intr-o lista
metrics = []

for factor in factors:
    print(f"\n🔁 Antrenez modelul pentru {factor}...")

    # Selecteaza oloanele pt acel factor
    X = df[[f'{factor}{i}' for i in range(1, 8)]].values
    y = df[[f'{factor}{i}' for i in range(8, 11)]].values

    # split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    # scale data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)
    y_test = scaler_y.transform(y_test)

    joblib.dump(scaler_X, f"scalers/scaler_X_{factor}.pkl")
    joblib.dump(scaler_y, f"scalers/scaler_y_{factor}.pkl")

    # Defineste modelul
    inputs = tf.keras.Input(shape=(7,))
    x = tf.keras.layers.Dense(128, activation='tanh')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(64, activation='tanh')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(32, activation='tanh')(x)
    outputs = tf.keras.layers.Dense(3)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss='mse')


    # train ze model
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=25,
                        batch_size=64,
                        verbose=1)

    # Evalueaza
    y_val_pred = scaler_y.inverse_transform(model.predict(X_val))
    y_test_pred = scaler_y.inverse_transform(model.predict(X_test))
    y_val_true = scaler_y.inverse_transform(y_val)
    y_test_true = scaler_y.inverse_transform(y_test)

    mse_val = mean_squared_error(y_val_true, y_val_pred)
    mse_test = mean_squared_error(y_test_true, y_test_pred)
    mae_val = mean_absolute_error(y_val_true, y_val_pred)
    mae_test = mean_absolute_error(y_test_true, y_test_pred)

    print(f"{factor} - MSE Test: {mse_test:.4f} | MAE Test: {mae_test:.4f}")

    # Save modelul
    model.save(f"models/model_{factor}.keras")

    # Salveaza loss graph
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"{factor} - Loss per Epoca")
    plt.xlabel("Epoca")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/loss_{factor}.png")
    plt.close()

    # Metrice
    metrics.append({
        'Factor': factor,
        'MSE_Val': mse_val,
        'MSE_Test': mse_test,
        'MAE_Val': mae_val,
        'MAE_Test': mae_test
    })

# Export metrice
pd.DataFrame(metrics).to_csv("metrics_summary.csv", index=False)
