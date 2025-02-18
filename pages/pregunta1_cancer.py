# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 11:37:44 2025

@author: jperezr
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Datos de entrenamiento
data = {
    'Temperatura': [36.5, 37.2, 36.8, 37.5, 36.9, 37.0, 35.3, 37.1, 34.9, 38.1, 36.7, 36.5, 37.22],
    'Presion_Arterial': [120, 140, 130, 150, 125, 145, 129, 141, 153, 151, 125, 158, 160],
    'Comorbilidad': ['Obesidad', 'Obesidad', 'Ansiedad', 'Asma', 'Asma', 'Arritmia', 'Ansiedad', 'Obesidad',
                     'Ansiedad', 'Obesidad', 'Ansiedad', 'Asma', 'Arritmia'],
    'Clasificacion': ['S', 'E', 'S', 'E', 'S', 'E', 'S', 'E', 'S', 'E', 'S', 'E', 'E']
}

# Crear el DataFrame
df = pd.DataFrame(data)

# Codificar variables categóricas
label_encoder_comorbilidad = LabelEncoder()
df['Comorbilidad'] = label_encoder_comorbilidad.fit_transform(df['Comorbilidad'])
label_encoder_clasificacion = LabelEncoder()
df['Clasificacion'] = label_encoder_clasificacion.fit_transform(df['Clasificacion'])

# Dividir las características (X) y el objetivo (y)
X = df[['Temperatura', 'Presion_Arterial', 'Comorbilidad']].values
y = df['Clasificacion'].values

# Implementación de Naive Bayes con EMV
class NaiveBayesEMV:
    def fit(self, X, y):
        self.classes, counts = np.unique(y, return_counts=True)
        self.priors = counts / len(y)
        self.feature_stats = {}

        for c in self.classes:
            X_c = X[y == c]
            self.feature_stats[c] = {
                'mean': X_c.mean(axis=0),
                'var': X_c.var(axis=0),
            }

    def _gaussian_prob(self, x, mean, var):
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return coeff * exponent

    def predict_proba(self, X):
        probabilities = []
        for x in X:
            class_probs = []
            for c in self.classes:
                prior = self.priors[c]
                likelihoods = np.prod(
                    self._gaussian_prob(x, self.feature_stats[c]['mean'], self.feature_stats[c]['var'])
                )
                class_probs.append(prior * likelihoods)
            total = np.sum(class_probs)
            normalized_probs = [p / total for p in class_probs]
            probabilities.append(normalized_probs)
        return np.array(probabilities)

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

# Entrenar el modelo personalizado
nb_emv = NaiveBayesEMV()
nb_emv.fit(X, y)

# Streamlit UI
st.title("Clasificación de Pacientes con Naive Bayes")
st.write("Este modelo clasifica a los pacientes como 'Sano' o 'Enfermo' basado en temperatura, presión arterial y comorbilidad.")

st.subheader("Datos de Entrenamiento")
st.dataframe(df)

# Entrada del usuario para Predicción Original
st.subheader("Ingresar Datos para Predicción (Original)")
temperatura = st.selectbox("Selecciona Temperatura (°C)", [36.5, 37.2, 36.8, 37.5, 36.9, 37.0, 35.3, 37.1, 34.9, 38.1, 36.7, 37.22])
presion = st.selectbox("Selecciona Presión Arterial (mmHg)", [120, 140, 130, 150, 125, 145, 129, 141, 153, 151, 125, 158, 160])
comorbilidad = st.selectbox("Selecciona Comorbilidad", label_encoder_comorbilidad.classes_)

# Predicción original
comorbilidad_encoded = label_encoder_comorbilidad.transform([comorbilidad])[0]
nuevo_paciente = np.array([[temperatura, presion, comorbilidad_encoded]])

probas_original = nb_emv.predict_proba(nuevo_paciente)[0]
prediccion_original = nb_emv.predict(nuevo_paciente)[0]
clase_predicha_original = label_encoder_clasificacion.inverse_transform([prediccion_original])[0]

# Entrada del usuario para Predicción con Ruido (Comorbilidad: 'Cáncer')
st.subheader("Ingresar Datos para Predicción con Ruido o Información Irrelevante (Comorbilidad: 'Cáncer')")
temperatura_irrelevante = st.selectbox("Selecciona Temperatura (°C) para Ruido", [36.5, 37.2, 36.8, 37.5, 36.9, 37.0, 35.3, 37.1, 34.9, 38.1, 36.7, 37.22])
presion_irrelevante = st.selectbox("Selecciona Presión Arterial (mmHg) para Ruido", [120, 140, 130, 150, 125, 145, 129, 141, 153, 151, 125, 158, 160])

# Aquí agregamos la opción para seleccionar comorbilidad, incluyendo "Cáncer"
comorbilidad_irrelevante = st.selectbox("Selecciona Comorbilidad (incluyendo 'Cáncer')", label_encoder_comorbilidad.classes_.tolist() + ['Cáncer'])

# Verificamos si la comorbilidad es reconocida por el encoder
if comorbilidad_irrelevante in label_encoder_comorbilidad.classes_:
    comorbilidad_encoded_irrelevante = label_encoder_comorbilidad.transform([comorbilidad_irrelevante])[0]
else:
    # Si no está en las clases, asignamos un valor por defecto para 'Cáncer'
    comorbilidad_encoded_irrelevante = -1  # Valor fuera del rango esperado

# Crear un paciente con estos cambios
nuevo_paciente_con_ruido = np.array([[temperatura_irrelevante, presion_irrelevante, comorbilidad_encoded_irrelevante]])

# Predicción con datos alterados
probas_con_ruido = nb_emv.predict_proba(nuevo_paciente_con_ruido)[0]
prediccion_con_ruido = nb_emv.predict(nuevo_paciente_con_ruido)[0]
clase_predicha_con_ruido = label_encoder_clasificacion.inverse_transform([prediccion_con_ruido])[0]

# Mostrar resultados con y sin ruido
st.subheader("Resultados con y sin Ruido")
st.write(f"### Predicción Original (Con Datos Correctos):")
st.write(f"Clase Predicha Original: **{clase_predicha_original}**")
st.write(f"Probabilidad de Enfermo (E): {probas_original[0]:.4f}")
st.write(f"Probabilidad de Sano (S): {probas_original[1]:.4f}")

st.write(f"### Resultados con Ruido o Información Irrelevante (Comorbilidad: 'Cáncer'):")
st.write(f"Comorbilidad Introducida: **{comorbilidad_irrelevante}**")
st.write(f"Clase Predicha con Ruido o Información Irrelevante: **{clase_predicha_con_ruido}**")
st.write(f"Probabilidad de Enfermo (E): {probas_con_ruido[0]:.4f}")
st.write(f"Probabilidad de Sano (S): {probas_con_ruido[1]:.4f}")
