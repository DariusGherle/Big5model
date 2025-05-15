import numpy as np
import joblib
import tensorflow as tf

def predict(factor, inputs):
    assert len(inputs) == 7, f"{factor} trebuie sa aiba 7 val"

    # load odel si scalerele
    model = tf.keras.models.load_model(f"models/model_{factor}.keras")
    scaler_X = joblib.load(f"scalers/scaler_X_{factor}.pkl")
    scaler_y = joblib.load(f"scalers/scaler_y_{factor}.pkl")

    # transfo input and predit
    x_scaled = scaler_X.transform(np.array(inputs).reshape(1, -1))
    y_scaled = model.predict(x_scaled)
    y = scaler_y.inverse_transform(y_scaled)

    return y.flatten()

# Test direct cu date hardcodate
test_inputs = {
    "EXT": [3, 2, 4, 4, 3, 3, 4],
    "EST": [2, 2, 4, 4, 4, 3, 2],
    "AGR": [3, 4, 2, 4, 4, 3, 3],
    "CSN": [2, 4, 4, 4, 4, 3, 2],
    "OPN": [3, 2, 2, 4, 3, 3, 3]
}

# Afisare rezultate
for factor, inputs in test_inputs.items():
    print(f"\n🔎 {factor} → input: {inputs}")
    preds = predict(factor, inputs)
    print("Prd:", [f"{factor}{i+8}: {val:.2f}" for i, val in enumerate(preds)])
