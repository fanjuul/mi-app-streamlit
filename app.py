import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Título de la app
st.title("Visualización de Regresión Lineal")

# Crear datos de ejemplo
st.write("Generando datos de ejemplo...")
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Mostrar los primeros datos
st.write(pd.DataFrame(np.hstack([X, y]), columns=["X", "y"]))

# Entrenar modelo
model = LinearRegression()
model.fit(X, y)

# Predicciones
X_new = np.array([[0], [2]])
y_pred = model.predict(X_new)

# Mostrar coeficientes
st.write(f"Coeficiente: {model.coef_[0][0]:.2f}")
st.write(f"Intercepto: {model.intercept_[0]:.2f}")

# Visualización
fig, ax = plt.subplots()
ax.scatter(X, y, color="blue", label="Datos reales")
ax.plot(X_new, y_pred, color="red", linewidth=2, label="Predicción")
ax.set_xlabel("X")
ax.set_ylabel("y")
ax.legend()
st.pyplot(fig)
