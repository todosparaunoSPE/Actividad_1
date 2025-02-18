# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:55:33 2025

@author: jperezr
"""

import streamlit as st

# Título de la aplicación
st.title("Información")

# Datos
estudiante = "Javier Horacio Pérez Ricárdez"
asignatura = "Aprendizaje M. Aplicado"
profesor = "Omar Velázquez López"
actividad = "Aplicación del Clasificador Bayesiano Ingenuo Usando EMV"
fecha = "Febrero del 2025"
bibliografia = """
Velázquez, O. (2025). *Aprendizaje automático aplicado en la clasificación de datos*. Editorial XYZ.
"""

# Configurar la barra lateral
st.sidebar.title("Información")
st.sidebar.write("**Estudiante:** Javier Horacio Pérez Ricárdez")
st.sidebar.write("© 2025 Javier Horacio Pérez Ricárdez. Todos los derechos reservados.")


# Botón para descargar el archivo PDF
with open("modelo_NB_Ingenuo_EMV.pdf", "rb") as file:
    btn = st.sidebar.download_button(
        label="📥 Descargar Modelo NB Ingenuo",
        data=file,
        file_name="modelo_NB_Ingenuo_EMV.pdf",
        mime="application/pdf"
    )


# Mostrar los datos
st.header("Detalles del Estudiante")
st.write(f"**Estudiante:** {estudiante}")
st.write(f"**Asignatura:** {asignatura}")
st.write(f"**Profesor:** {profesor}")
st.write(f"**Actividad:** {actividad}")
st.write(f"**Fecha:** {fecha}")
#st.write(f"**Bibliografía (APA):** {bibliografia}")
