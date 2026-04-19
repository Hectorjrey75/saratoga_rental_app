"""
Página de Predicción por Lote.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import io

from src.config.settings import config, MODELS_DIR, RAW_DATA_DIR
from src.prediction.prediction import RentalPricePredictor
from src.utils.helpers import format_currency
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Predicción por Lote - Rental Price Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Predicción por Lote")
st.markdown("---")


@st.cache_resource
def load_predictor():
    """Cargar el modelo entrenado."""
    model_path = MODELS_DIR / "best_model.pkl"
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    
    if model_path.exists() and preprocessor_path.exists():
        try:
            predictor = RentalPricePredictor(model_path, preprocessor_path)
            logger.info("Modelo cargado exitosamente")
            return predictor
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            return None
    return None


@st.cache_data
def load_sample_data():
    """Cargar datos de ejemplo."""
    data_path = RAW_DATA_DIR / "SaratogaHouses.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        # Eliminar columna price si existe (para predicción)
        if 'price' in df.columns:
            df = df.drop('price', axis=1)
        return df.head(20)
    return None


def create_template_csv():
    """Crear plantilla CSV para descarga."""
    template_data = {
        "lotSize": [0.5, 0.8],
        "age": [15, 8],
        "landValue": [50000, 75000],
        "livingArea": [1500, 2200],
        "pctCollege": [50, 65],
        "bedrooms": [3, 4],
        "fireplaces": [1, 2],
        "bathrooms": [2.0, 2.5],
        "rooms": [6, 8],
        "heating": ["hot air", "hot air"],
        "fuel": ["gas", "gas"],
        "sewer": ["septic", "public/commercial"],
        "waterfront": ["No", "No"],
        "newConstruction": ["No", "Yes"],
        "centralAir": ["No", "Yes"]
    }
    return pd.DataFrame(template_data)


def validate_data(df):
    """Validar que el DataFrame tenga las columnas requeridas."""
    required_columns = [
        "lotSize", "age", "landValue", "livingArea", "pctCollege",
        "bedrooms", "fireplaces", "bathrooms", "rooms",
        "heating", "fuel", "sewer", "waterfront", "newConstruction", "centralAir"
    ]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        return False, f"Faltan columnas requeridas: {missing_cols}"
    
    return True, "Datos válidos"


def display_prediction_results(results_df, original_df):
    """Mostrar resultados de predicciones."""
    
    # Métricas resumen
    st.markdown("### 📊 Resumen de Predicciones")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Propiedades", len(results_df))
    with col2:
        st.metric("Precio Promedio", format_currency(results_df['predicted_price'].mean()))
    with col3:
        st.metric("Precio Mediano", format_currency(results_df['predicted_price'].median()))
    with col4:
        st.metric("Precio Mínimo", format_currency(results_df['predicted_price'].min()))
    with col5:
        st.metric("Precio Máximo", format_currency(results_df['predicted_price'].max()))
    
    # Gráficos de distribución
    st.markdown("### 📈 Distribución de Precios Predichos")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Histograma
        fig_hist = px.histogram(
            results_df, x='predicted_price',
            nbins=30,
            title='Histograma de Precios Predichos',
            labels={'predicted_price': 'Precio Predicho ($)'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col_chart2:
        # Boxplot
        fig_box = px.box(
            results_df, y='predicted_price',
            title='Boxplot de Precios Predichos',
            labels={'predicted_price': 'Precio Predicho ($)'},
            color_discrete_sequence=['#ff7f0e']
        )
        fig_box.update_layout(height=400)
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Tabla de resultados
    st.markdown("### 📋 Tabla de Resultados")
    
    # Formatear columnas numéricas
    display_df = results_df.copy()
    if 'predicted_price' in display_df.columns:
        display_df['Precio Predicho'] = display_df['predicted_price'].apply(format_currency)
    if 'ci_lower' in display_df.columns:
        display_df['Límite Inferior'] = display_df['ci_lower'].apply(format_currency)
    if 'ci_upper' in display_df.columns:
        display_df['Límite Superior'] = display_df['ci_upper'].apply(format_currency)
    
    # Seleccionar columnas para mostrar
    display_cols = ['Precio Predicho']
    if 'Límite Inferior' in display_df.columns:
        display_cols.extend(['Límite Inferior', 'Límite Superior'])
    
    # Añadir columnas originales relevantes
    relevant_cols = ['livingArea', 'bedrooms', 'bathrooms', 'age', 'waterfront', 'centralAir']
    for col in relevant_cols:
        if col in display_df.columns:
            display_cols.append(col)
    
    st.dataframe(display_df[display_cols], use_container_width=True)
    
    # Análisis por características
    st.markdown("### 🔍 Análisis por Características")
    
    col_analysis1, col_analysis2 = st.columns(2)
    
    with col_analysis1:
        # Precio por dormitorios
        if 'bedrooms' in results_df.columns:
            bedroom_prices = results_df.groupby('bedrooms')['predicted_price'].mean().reset_index()
            fig_bed = px.bar(
                bedroom_prices, x='bedrooms', y='predicted_price',
                title='Precio Promedio por Número de Dormitorios',
                labels={'bedrooms': 'Dormitorios', 'predicted_price': 'Precio Promedio ($)'},
                text=bedroom_prices['predicted_price'].apply(lambda x: format_currency(x)),
                color='predicted_price',
                color_continuous_scale='Viridis'
            )
            fig_bed.update_traces(textposition='outside')
            fig_bed.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_bed, use_container_width=True)
    
    with col_analysis2:
        # Precio por área habitable (categorizada)
        if 'livingArea' in results_df:
            results_df['area_category'] = pd.cut(
                results_df['livingArea'],
                bins=[0, 1000, 1500, 2000, 3000, 10000],
                labels=['<1000', '1000-1500', '1500-2000', '2000-3000', '3000+']
            )
            area_prices = results_df.groupby('area_category', observed=True)['predicted_price'].mean().reset_index()
            fig_area = px.bar(
                area_prices, x='area_category', y='predicted_price',
                title='Precio Promedio por Categoría de Área',
                labels={'area_category': 'Área (m²)', 'predicted_price': 'Precio Promedio ($)'},
                text=area_prices['predicted_price'].apply(lambda x: format_currency(x)),
                color='predicted_price',
                color_continuous_scale='Viridis'
            )
            fig_area.update_traces(textposition='outside')
            fig_area.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_area, use_container_width=True)
    
    # Scatter plot: Área vs Precio
    if 'livingArea' in results_df.columns:
        st.markdown("### 📈 Relación Área vs Precio")
        
        fig_scatter = px.scatter(
            results_df, x='livingArea', y='predicted_price',
            color='bedrooms' if 'bedrooms' in results_df.columns else None,
            size='bathrooms' if 'bathrooms' in results_df.columns else None,
            hover_data=['age', 'waterfront'] if 'waterfront' in results_df.columns else ['age'],
            title='Precio Predicho vs Área Habitable',
            labels={'livingArea': 'Área Habitable (m²)', 'predicted_price': 'Precio Predicho ($)'},
            opacity=0.7
        )
        
        # Añadir línea de tendencia
        z = np.polyfit(results_df['livingArea'], results_df['predicted_price'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(results_df['livingArea'].min(), results_df['livingArea'].max(), 100)
        
        fig_scatter.add_trace(
            go.Scatter(x=x_line, y=p(x_line), mode='lines',
                       name='Tendencia', line=dict(color='red', width=2, dash='dash'))
        )
        
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    return results_df


# Cargar predictor
predictor = load_predictor()

if predictor is None:
    st.warning("⚠️ No hay modelo entrenado. Por favor, entrena el modelo desde la página principal.")
else:
    # Opciones de entrada
    st.markdown("### 📤 Método de Entrada")
    
    input_method = st.radio(
        "Selecciona cómo quieres ingresar los datos",
        ["📁 Subir Archivo CSV", "📋 Usar Datos de Ejemplo", "✏️ Entrada Manual"],
        horizontal=True
    )
    
    data_to_predict = None
    
    if input_method == "📁 Subir Archivo CSV":
        st.markdown("#### Subir Archivo CSV")
        
        # Descargar plantilla
        template_df = create_template_csv()
        csv_template = template_df.to_csv(index=False)
        
        st.download_button(
            "📥 Descargar Plantilla CSV",
            csv_template,
            "plantilla_prediccion.csv",
            "text/csv",
            help="Descarga una plantilla con el formato correcto"
        )
        
        uploaded_file = st.file_uploader(
            "Selecciona un archivo CSV con las propiedades a predecir",
            type=["csv"],
            help="El archivo debe contener todas las columnas requeridas"
        )
        
        if uploaded_file is not None:
            try:
                data_to_predict = pd.read_csv(uploaded_file)
                st.success(f"✅ Archivo cargado: {len(data_to_predict)} propiedades")
                
                # Validar datos
                is_valid, message = validate_data(data_to_predict)
                if not is_valid:
                    st.error(f"❌ {message}")
                    data_to_predict = None
                else:
                    st.dataframe(data_to_predict.head(10), use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error al leer el archivo: {e}")
    
    elif input_method == "📋 Usar Datos de Ejemplo":
        st.markdown("#### Datos de Ejemplo")
        
        sample_data = load_sample_data()
        if sample_data is not None:
            data_to_predict = sample_data
            st.success(f"✅ Datos de ejemplo cargados: {len(data_to_predict)} propiedades")
            st.dataframe(data_to_predict, use_container_width=True)
        else:
            st.error("❌ No se pudieron cargar los datos de ejemplo")
    
    elif input_method == "✏️ Entrada Manual":
        st.markdown("#### Entrada Manual de Propiedades")
        
        n_properties = st.number_input(
            "Número de propiedades a ingresar",
            min_value=1, max_value=20, value=3
        )
        
        manual_data = []
        
        for i in range(n_properties):
            with st.expander(f"Propiedad {i+1}", expanded=(i == 0)):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    lot_size = st.number_input(f"Tamaño Terreno", 0.0, 20.0, 0.5, 0.1, key=f"lot_{i}")
                    age = st.number_input(f"Antigüedad", 0, 300, 15, 1, key=f"age_{i}")
                    land_value = st.number_input(f"Valor Terreno", 0, 500000, 50000, 1000, key=f"land_{i}")
                    living_area = st.number_input(f"Área Habitable", 100, 10000, 1500, 50, key=f"living_{i}")
                    pct_college = st.slider(f"% Universitario", 0, 100, 50, 1, key=f"pct_{i}")
                
                with col2:
                    bedrooms = st.number_input(f"Dormitorios", 0, 10, 3, 1, key=f"bed_{i}")
                    bathrooms = st.number_input(f"Baños", 0.0, 8.0, 2.0, 0.5, key=f"bath_{i}")
                    fireplaces = st.number_input(f"Chimeneas", 0, 5, 1, 1, key=f"fire_{i}")
                    rooms = st.number_input(f"Habitaciones", 1, 20, 6, 1, key=f"rooms_{i}")
                
                with col3:
                    heating = st.selectbox("Calefacción", ["hot air", "hot water/steam", "electric"], key=f"heat_{i}")
                    fuel = st.selectbox("Combustible", ["gas", "oil", "electric"], key=f"fuel_{i}")
                    sewer = st.selectbox("Desagüe", ["septic", "public/commercial", "none"], key=f"sewer_{i}")
                    waterfront = st.selectbox("Vista al Lago", ["No", "Yes"], key=f"water_{i}")
                    new_const = st.selectbox("Nueva Construcción", ["No", "Yes"], key=f"new_{i}")
                    central_air = st.selectbox("Aire Acondicionado", ["No", "Yes"], key=f"air_{i}")
                
                manual_data.append({
                    "lotSize": lot_size, "age": age, "landValue": land_value,
                    "livingArea": living_area, "pctCollege": pct_college,
                    "bedrooms": bedrooms, "fireplaces": fireplaces,
                    "bathrooms": bathrooms, "rooms": rooms,
                    "heating": heating, "fuel": fuel, "sewer": sewer,
                    "waterfront": waterfront, "newConstruction": new_const,
                    "centralAir": central_air
                })
        
        if manual_data:
            data_to_predict = pd.DataFrame(manual_data)
    
    # Botón de predicción
    if data_to_predict is not None:
        st.markdown("---")
        
        if st.button("🚀 Realizar Predicciones", type="primary", use_container_width=True):
            with st.spinner("🔮 Procesando predicciones..."):
                try:
                    results = predictor.predict_batch(data_to_predict, return_confidence=True)
                    
                    # Mostrar resultados
                    display_prediction_results(results, data_to_predict)
                    
                    # Opciones de descarga
                    st.markdown("### 📥 Descargar Resultados")
                    
                    col_down1, col_down2 = st.columns(2)
                    
                    with col_down1:
                        # Descargar resultados completos
                        csv_results = results.to_csv(index=False)
                        st.download_button(
                            "📊 Descargar Resultados Completos (CSV)",
                            csv_results,
                            f"predicciones_lote_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    
                    with col_down2:
                        # Descargar resumen
                        summary = pd.DataFrame({
                            'Métrica': ['Total Propiedades', 'Precio Promedio', 'Precio Mediano', 
                                       'Precio Mínimo', 'Precio Máximo', 'Desviación Estándar'],
                            'Valor': [
                                len(results),
                                format_currency(results['predicted_price'].mean()),
                                format_currency(results['predicted_price'].median()),
                                format_currency(results['predicted_price'].min()),
                                format_currency(results['predicted_price'].max()),
                                format_currency(results['predicted_price'].std())
                            ]
                        })
                        csv_summary = summary.to_csv(index=False)
                        st.download_button(
                            "📋 Descargar Resumen (CSV)",
                            csv_summary,
                            f"resumen_predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    
                except Exception as e:
                    st.error(f"❌ Error al realizar predicciones: {e}")
                    logger.error(f"Error en predicción por lote: {e}")
    
    # Información adicional
    with st.expander("ℹ️ Información sobre Predicción por Lote"):
        st.markdown("""
        ### 📊 Predicción por Lote
        
        Esta funcionalidad permite predecir el precio de múltiples propiedades simultáneamente.
        
        **Formatos de entrada aceptados:**
        - **CSV**: Archivo con todas las columnas requeridas
        - **Datos de ejemplo**: Muestra de propiedades del dataset original
        - **Entrada manual**: Ingresar propiedades una por una
        
        **Columnas requeridas:**
        - `lotSize`: Tamaño del terreno en acres
        - `age`: Antigüedad en años
        - `landValue`: Valor del terreno en dólares
        - `livingArea`: Área habitable en m²
        - `pctCollege`: Porcentaje universitario del vecindario
        - `bedrooms`: Número de dormitorios
        - `fireplaces`: Número de chimeneas
        - `bathrooms`: Número de baños
        - `rooms`: Número total de habitaciones
        - `heating`: Tipo de calefacción
        - `fuel`: Tipo de combustible
        - `sewer`: Tipo de desagüe
        - `waterfront`: Vista al lago (Yes/No)
        - `newConstruction`: Nueva construcción (Yes/No)
        - `centralAir`: Aire acondicionado central (Yes/No)
        
        **Resultados generados:**
        - Precio predicho para cada propiedad
        - Intervalos de confianza del 95%
        - Análisis estadístico del lote
        - Visualizaciones comparativas
        """)