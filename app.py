import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("Regresión Lineal Simple")

# Subir archivo CSV
file = st.file_uploader("Sube tu CSV", type="csv")

if file:
    df = pd.read_csv(file)
    st.write(df)

    # Selección de columnas
    x_col = st.selectbox("Variable independiente (X)", df.columns)
    y_col = st.selectbox("Variable dependiente (Y)", df.columns)

    X = df[[x_col]]
    y = df[y_col]

    # Entrenar modelo
    model = LinearRegression()
    model.fit(X, y)

    st.write(f"Pendiente: {model.coef_[0]}")
    st.write(f"Intersección: {model.intercept_}")

    # Graficar los datos y la línea de regresión
    plt.scatter(X, y, color='blue', label='Datos')
    plt.plot(X, model.predict(X), color='red', label='Regresión')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("Visualización del Modelo")
    plt.legend()
    st.pyplot(plt)

    # Predecir nuevo valor
    new_x = st.number_input(f"Ingrese nuevo valor de {x_col}")
    if st.button("Predecir"):
        pred = model.predict(pd.DataFrame({x_col: [new_x]}))
        st.write(f"Predicción de {y_col}: {pred[0]}")
