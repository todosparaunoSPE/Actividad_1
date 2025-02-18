# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:55:33 2025

@author: jperezr
"""

import streamlit as st

# Título de la aplicación
st.title("Información del Estudiante y Actividad")

# Datos
estudiante = "Javier Horacio Pérez Ricárdez"
asignatura = "Aprendizaje M. Aplicado"
profesor = "Omar Velázquez López"
actividad = "Aplicación del Clasificador Bayesiano Ingenuo"
fecha = "Febrero del 2025"
bibliografia = """
Velázquez, O. (2025). *Aprendizaje automático aplicado en la clasificación de datos*. Editorial XYZ.
"""

# Mostrar los datos
st.header("Detalles del Estudiante")
st.write(f"**Estudiante:** {estudiante}")
st.write(f"**Asignatura:** {asignatura}")
st.write(f"**Profesor:** {profesor}")
st.write(f"**Actividad:** {actividad}")
st.write(f"**Fecha:** {fecha}")
st.write(f"**Bibliografía (APA):** {bibliografia}")
