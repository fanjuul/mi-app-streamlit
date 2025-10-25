import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

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

    # Predecir nuevo valor
    new_x = st.number_input(f"Ingrese nuevo valor de {x_col}")
    if st.button("Predecir"):
        # Crear DataFrame con el mismo nombre de columna
        pred = model.predict(pd.DataFrame({x_col: [new_x]}))
        st.write(f"Predicción de {y_col}: {pred[0]}")

