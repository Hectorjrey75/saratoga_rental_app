# 🏠 Sistema Profesional de Predicción de Precios de Alquiler

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-AA4A44.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Sistema profesional de predicción de precios de alquiler de viviendas utilizando técnicas avanzadas de Machine Learning, con una interfaz web interactiva desarrollada en Streamlit.

## 📋 Tabla de Contenidos

- [🏠 Sistema Profesional de Predicción de Precios de Alquiler](#-sistema-profesional-de-predicción-de-precios-de-alquiler)
  - [📋 Tabla de Contenidos](#-tabla-de-contenidos)
  - [✨ Características](#-características)
  - [📦 Requisitos](#-requisitos)
  - [🔧 Instalación](#-instalación)
    - [Instalación Local](#instalación-local)
- [Clonar el repositorio](#clonar-el-repositorio)
- [Crear entorno virtual](#crear-entorno-virtual)
- [Instalar dependencias](#instalar-dependencias)
- [Estructura del Proyecto](#estructura-del-proyecto)

## ✨ Características

- **Predicción Individual**: Estimación precisa para propiedades específicas
- **Predicción por Lote**: Procesamiento simultáneo de múltiples propiedades
- **Análisis de Escenarios**: Comparación de diferentes configuraciones
- **Forecast Recursivo**: Proyecciones multietapa con intervalos de confianza
- **Análisis de Sensibilidad**: Evaluación del impacto de cada variable
- **Dashboard Interactivo**: Visualizaciones avanzadas con Plotly
- **Registro de Eventos**: Sistema completo de logging
- **Gestión de Configuración**: Configuración centralizada y flexible
- **Código Profesional**: Estructura limpia, modular y mantenible

## 📦 Requisitos

- Python 3.10+
- Dependencias listadas en `requirements.txt`

## 🔧 Instalación

### Instalación Local

```bash
# Clonar el repositorio
git clone https://github.com/yourusername/rental_price_prediction.git
cd rental_price_prediction

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
pip install -e .

# Estructura del Proyecto
Saratoga_Houses/
├── data/
│   ├── raw/
│   │   └── SaratogaHouses.csv
│   ├── processed/
│   │   ├── eda_processed_data.csv
│   │   └── eda_summary.csv
│   └── models/
│       ├── best_model.pkl
│       └── preprocessor.pkl
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── config/
│   │   └── settings.py
│   ├── data/
│   │   └── preprocessing.py
│   ├── models/
│   │   └── model_training.py
│   ├── prediction/
│   │   └── prediction.py
│   └── utils/
│       ├── logger.py
│       └── helpers.py
├── app/
│   ├── main.py
│   └── pages/
│       ├── __init__.py
│       ├── 1_📊_Analisis_Exploratorio.py
│       ├── 2_🔮_Prediccion_Individual.py
│       ├── 3_📈_Prediccion_Lote.py
│       └── 4_🎯_Analisis_Escenarios.py
├── logs/
├── train_model.py
├── requirements.txt
└── README.md