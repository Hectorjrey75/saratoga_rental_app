"""
Página de Predicción Individual.
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
from datetime import datetime

from src.config.settings import config, MODELS_DIR
from src.prediction.prediction import RentalPricePredictor
from src.utils.helpers import format_currency
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Predicción Individual - Rental Price Predictor",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Predicción Individual de Precio")
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
            st.error(f"Error al cargar el modelo: {e}")
            return None
    else:
        logger.warning("No se encontró modelo entrenado")
        return None


def create_gauge_chart(value, title="Precio Predicho"):
    """Crear gráfico de gauge para visualizar el precio."""
    # Normalizar valor para el gauge (asumiendo rango 0-1,000,000)
    max_val = 1000000
    normalized = min(value / max_val, 1.0)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title, 'font': {'size': 20}},
        number={'prefix': "$", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, max_val], 'tickprefix': "$", 'tickformat': ',d'},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 200000], 'color': "#e74c3c"},
                {'range': [200000, 400000], 'color': "#f1c40f"},
                {'range': [400000, 600000], 'color': "#2ecc71"},
                {'range': [600000, 800000], 'color': "#3498db"},
                {'range': [800000, max_val], 'color': "#9b59b6"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def display_feature_impact(predictor, features):
    """Mostrar impacto de características."""
    if predictor.get_feature_importance() is not None:
        importance_df = predictor.get_feature_importance().head(10)
        
        fig = px.bar(
            importance_df, x='importance', y='feature',
            orientation='h',
            title='Top 10 Características Más Influyentes',
            labels={'importance': 'Importancia', 'feature': 'Característica'},
            color='importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def display_prediction_history():
    """Mostrar historial de predicciones."""
    if 'prediction_history' in st.session_state and st.session_state.prediction_history:
        st.markdown("### 📜 Historial de Predicciones")
        
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df, use_container_width=True)
        
        if len(history_df) > 1:
            fig = px.line(
                history_df, y='predicted_price', 
                title='Tendencia de Predicciones',
                labels={'index': 'Predicción', 'predicted_price': 'Precio'}
            )
            fig.update_traces(mode='lines+markers')
            st.plotly_chart(fig, use_container_width=True)
        
        if st.button("🗑️ Limpiar Historial"):
            st.session_state.prediction_history = []
            st.rerun()


def generate_insights(features, prediction):
    """Generar insights basados en las características."""
    insights = []
    
    if features.get('waterfront') == 'Yes':
        insights.append("🌊 La propiedad tiene vista al lago, lo que aumenta significativamente su valor.")
    
    if features.get('centralAir') == 'Yes':
        insights.append("❄️ El aire acondicionado central añade valor a la propiedad.")
    
    if features.get('newConstruction') == 'Yes':
        insights.append("🏗️ Al ser nueva construcción, la propiedad tiene un valor premium.")
    
    if features.get('fireplaces', 0) >= 2:
        insights.append(f"🔥 Con {features['fireplaces']} chimeneas, la propiedad es más atractiva.")
    
    if features.get('pctCollege', 0) > 60:
        insights.append(f"🎓 El vecindario tiene un alto porcentaje universitario ({features['pctCollege']}%).")
    
    if features.get('age', 100) < 10:
        insights.append(f"📅 La propiedad es relativamente nueva ({features['age']} años).")
    elif features.get('age', 0) > 50:
        insights.append(f"🏚️ La propiedad tiene {features['age']} años, lo que puede requerir mantenimiento.")
    
    if features.get('livingArea', 0) > 2500:
        insights.append(f"🏠 Con {features['livingArea']:.0f} m², es una propiedad espaciosa.")
    
    if features.get('bathrooms', 0) >= 3:
        insights.append(f"🚿 {features['bathrooms']} baños es un número superior al promedio.")
    
    return insights


# Cargar predictor
predictor = load_predictor()

if predictor is None:
    st.warning("⚠️ No hay modelo entrenado. Por favor, entrena el modelo desde la página principal.")
    st.info("💡 Ve a la página 'Home' y haz clic en 'Train Model'")
else:
    # Inicializar historial en session state
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    # Layout de dos columnas
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("### 📝 Características de la Propiedad")
        
        with st.form("prediction_form"):
            # Método de entrada
            input_method = st.radio(
                "Método de entrada",
                ["📝 Formulario Manual", "📋 Ejemplo Predefinido"],
                horizontal=True
            )
            
            if input_method == "📋 Ejemplo Predefinido":
                example_type = st.selectbox(
                    "Tipo de propiedad",
                    ["🏠 Casa Promedio", "🏰 Casa de Lujo", "🏘️ Casa Económica", "🏡 Casa con Vista al Lago"]
                )
                
                # Ejemplos predefinidos
                examples = {
                    "🏠 Casa Promedio": {
                        "lotSize": 0.5, "age": 20, "landValue": 50000,
                        "livingArea": 1800, "pctCollege": 50,
                        "bedrooms": 3, "fireplaces": 1, "bathrooms": 2.0, "rooms": 6,
                        "heating": "hot air", "fuel": "gas", "sewer": "septic",
                        "waterfront": "No", "newConstruction": "No", "centralAir": "No"
                    },
                    "🏰 Casa de Lujo": {
                        "lotSize": 2.5, "age": 5, "landValue": 150000,
                        "livingArea": 3500, "pctCollege": 75,
                        "bedrooms": 5, "fireplaces": 2, "bathrooms": 4.0, "rooms": 12,
                        "heating": "hot air", "fuel": "gas", "sewer": "public/commercial",
                        "waterfront": "No", "newConstruction": "Yes", "centralAir": "Yes"
                    },
                    "🏘️ Casa Económica": {
                        "lotSize": 0.2, "age": 45, "landValue": 25000,
                        "livingArea": 1000, "pctCollege": 35,
                        "bedrooms": 2, "fireplaces": 0, "bathrooms": 1.0, "rooms": 4,
                        "heating": "electric", "fuel": "electric", "sewer": "public/commercial",
                        "waterfront": "No", "newConstruction": "No", "centralAir": "No"
                    },
                    "🏡 Casa con Vista al Lago": {
                        "lotSize": 1.2, "age": 15, "landValue": 200000,
                        "livingArea": 2500, "pctCollege": 60,
                        "bedrooms": 4, "fireplaces": 2, "bathrooms": 3.0, "rooms": 9,
                        "heating": "hot water/steam", "fuel": "gas", "sewer": "septic",
                        "waterfront": "Yes", "newConstruction": "No", "centralAir": "Yes"
                    }
                }
                
                default_features = examples[example_type]
                st.info(f"Usando valores predefinidos para: {example_type}")
            else:
                default_features = {
                    "lotSize": 0.5, "age": 15, "landValue": 50000,
                    "livingArea": 1500, "pctCollege": 50,
                    "bedrooms": 3, "fireplaces": 1, "bathrooms": 2.0, "rooms": 6,
                    "heating": "hot air", "fuel": "gas", "sewer": "septic",
                    "waterfront": "No", "newConstruction": "No", "centralAir": "No"
                }
            
            # Características de la propiedad
            st.markdown("#### 📐 Características del Terreno")
            col_a, col_b = st.columns(2)
            with col_a:
                lot_size = st.number_input(
                    "Tamaño del Terreno (acres)", 
                    min_value=0.0, max_value=20.0, 
                    value=float(default_features["lotSize"]), step=0.1,
                    help="Área total del terreno en acres"
                )
                land_value = st.number_input(
                    "Valor del Terreno ($)", 
                    min_value=0, max_value=500000, 
                    value=int(default_features["landValue"]), step=1000,
                    help="Valor tasado del terreno"
                )
            with col_b:
                age = st.number_input(
                    "Antigüedad (años)", 
                    min_value=0, max_value=300, 
                    value=int(default_features["age"]), step=1,
                    help="Años desde la construcción"
                )
                living_area = st.number_input(
                    "Área Habitable (m²)", 
                    min_value=100, max_value=10000, 
                    value=int(default_features["livingArea"]), step=50,
                    help="Metros cuadrados de espacio habitable"
                )
            
            st.markdown("#### 🏠 Características del Interior")
            col_c, col_d = st.columns(2)
            with col_c:
                bedrooms = st.number_input(
                    "Dormitorios", 
                    min_value=0, max_value=10, 
                    value=int(default_features["bedrooms"]), step=1
                )
                bathrooms = st.number_input(
                    "Baños", 
                    min_value=0.0, max_value=8.0, 
                    value=float(default_features["bathrooms"]), step=0.5,
                    help="0.5 = baño sin ducha"
                )
                fireplaces = st.number_input(
                    "Chimeneas", 
                    min_value=0, max_value=5, 
                    value=int(default_features["fireplaces"]), step=1
                )
            with col_d:
                rooms = st.number_input(
                    "Habitaciones Totales", 
                    min_value=1, max_value=20, 
                    value=int(default_features["rooms"]), step=1
                )
                pct_college = st.slider(
                    "Porcentaje Universitario (%)", 
                    0, 100, 
                    value=int(default_features["pctCollege"]), step=1,
                    help="Porcentaje de residentes del vecindario con título universitario"
                )
            
            st.markdown("#### 🔧 Servicios y Amenidades")
            col_e, col_f = st.columns(2)
            with col_e:
                heating_options = ["hot air", "hot water/steam", "electric"]
                heating = st.selectbox(
                    "Tipo de Calefacción", 
                    heating_options,
                    index=heating_options.index(default_features["heating"]) if default_features["heating"] in heating_options else 0
                )
                
                fuel_options = ["gas", "oil", "electric"]
                fuel = st.selectbox(
                    "Combustible", 
                    fuel_options,
                    index=fuel_options.index(default_features["fuel"]) if default_features["fuel"] in fuel_options else 0
                )
                
                sewer_options = ["septic", "public/commercial", "none"]
                sewer = st.selectbox(
                    "Tipo de Desagüe", 
                    sewer_options,
                    index=sewer_options.index(default_features["sewer"]) if default_features["sewer"] in sewer_options else 0
                )
            with col_f:
                waterfront = st.selectbox(
                    "Vista al Lago", 
                    ["No", "Yes"],
                    index=1 if default_features["waterfront"] == "Yes" else 0
                )
                new_construction = st.selectbox(
                    "Nueva Construcción", 
                    ["No", "Yes"],
                    index=1 if default_features["newConstruction"] == "Yes" else 0
                )
                central_air = st.selectbox(
                    "Aire Acondicionado Central", 
                    ["No", "Yes"],
                    index=1 if default_features["centralAir"] == "Yes" else 0
                )
            
            # Botón de predicción
            submitted = st.form_submit_button("🔮 Predecir Precio", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Resultado de la Predicción")
        
        if submitted:
            # Construir features
            features = {
                "lotSize": lot_size,
                "age": age,
                "landValue": land_value,
                "livingArea": living_area,
                "pctCollege": pct_college,
                "bedrooms": bedrooms,
                "fireplaces": fireplaces,
                "bathrooms": bathrooms,
                "rooms": rooms,
                "heating": heating,
                "fuel": fuel,
                "sewer": sewer,
                "waterfront": waterfront,
                "newConstruction": new_construction,
                "centralAir": central_air
            }
            
            with st.spinner("🔮 Calculando predicción..."):
                try:
                    result = predictor.predict_single(features)
                    
                    # Guardar en historial
                    history_entry = {
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'predicted_price': result['predicted_price'],
                        'living_area': living_area,
                        'bedrooms': bedrooms,
                        'bathrooms': bathrooms,
                        'age': age
                    }
                    st.session_state.prediction_history.append(history_entry)
                    
                    # Mostrar resultado principal
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 20px;">
                        <h3 style="color: white; margin-bottom: 10px;">Precio Estimado de Alquiler</h3>
                        <h1 style="color: white; font-size: 52px;">{result['predicted_price_formatted']}</h1>
                        <p style="color: rgba(255,255,255,0.95); font-size: 16px;">
                            Intervalo de Confianza (95%):<br>
                            {result['confidence_interval']['lower_formatted']} - {result['confidence_interval']['upper_formatted']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Gauge chart
                    fig_gauge = create_gauge_chart(result['predicted_price'])
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Métricas adicionales
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        price_per_sqft = result['predicted_price'] / living_area if living_area > 0 else 0
                        st.metric("Precio por m²", format_currency(price_per_sqft))
                    with col_m2:
                        st.metric("Área Habitable", f"{living_area:,.0f} m²")
                    with col_m3:
                        uncertainty = (result['confidence_interval']['upper'] - result['confidence_interval']['lower']) / 2
                        st.metric("Incertidumbre (±)", format_currency(uncertainty))
                    
                    # Insights
                    st.markdown("### 💡 Insights")
                    insights = generate_insights(features, result['predicted_price'])
                    if insights:
                        for insight in insights:
                            st.info(insight)
                    else:
                        st.info("Esta es una propiedad con características promedio.")
                    
                    # Comparación con propiedades similares (VERSIÓN CORREGIDA)
                    st.markdown("### 📈 Comparación de Características")
                    
                    # Datos para el gráfico
                    categories = ['Área (m²)', 'Dormitorios', 'Baños', 'Antigüedad']
                    property_values = [
                        float(living_area), 
                        float(bedrooms), 
                        float(bathrooms), 
                        float(age)
                    ]
                    market_values = [1800.0, 3.2, 2.1, 35.0]
                    
                    # Formatear texto para las barras
                    property_text = [
                        f'{living_area:,.0f}',
                        f'{bedrooms:.0f}',
                        f'{bathrooms:.1f}',
                        f'{age:.0f}'
                    ]
                    market_text = [
                        f'{1800:,.0f}',
                        f'{3.2:.1f}',
                        f'{2.1:.1f}',
                        f'{35:.0f}'
                    ]
                    
                    # Crear figura con go.Figure
                    fig_comp = go.Figure(data=[
                        go.Bar(
                            name='Esta Propiedad', 
                            x=categories, 
                            y=property_values,
                            text=property_text, 
                            textposition='outside',
                            marker_color='#1f77b4',
                            hovertemplate='<b>%{x}</b><br>Esta Propiedad: %{text}<extra></extra>'
                        ),
                        go.Bar(
                            name='Promedio Mercado', 
                            x=categories, 
                            y=market_values,
                            text=market_text, 
                            textposition='outside',
                            marker_color='#95a5a6',
                            hovertemplate='<b>%{x}</b><br>Promedio Mercado: %{text}<extra></extra>'
                        )
                    ])
                    
                    fig_comp.update_layout(
                        title='Comparación con Promedios del Mercado',
                        height=400,
                        barmode='group',
                        yaxis_title='Valor',
                        xaxis_title='Característica',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5
                        ),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig_comp, use_container_width=True)
                    
                    # Tabla comparativa adicional
                    with st.expander("📊 Ver Tabla Comparativa Detallada"):
                        comp_table = pd.DataFrame({
                            'Característica': categories,
                            'Esta Propiedad': property_text,
                            'Promedio Mercado': market_text,
                            'Diferencia': [
                                f"{((living_area - 1800) / 1800 * 100):+.1f}%" if living_area > 0 else "N/A",
                                f"{((bedrooms - 3.2) / 3.2 * 100):+.1f}%",
                                f"{((bathrooms - 2.1) / 2.1 * 100):+.1f}%",
                                f"{((age - 35) / 35 * 100):+.1f}%" if age > 0 else "N/A"
                            ]
                        })
                        st.dataframe(comp_table, use_container_width=True, hide_index=True)
                    
                except Exception as e:
                    st.error(f"❌ Error al realizar la predicción: {e}")
                    logger.error(f"Error en predicción: {e}")
        else:
            # Mostrar información antes de predecir
            st.info("👆 Completa el formulario y haz clic en 'Predecir Precio' para obtener una estimación.")
            
            # Mostrar feature importance del modelo
            if predictor.get_feature_importance() is not None:
                display_feature_impact(predictor, {})
    
    # Sección de información adicional
    st.markdown("---")
    
    col_info1, col_info2 = st.columns([2, 1])
    
    with col_info1:
        st.markdown("### 📊 Métricas del Modelo")
        metrics = predictor.get_model_metrics()
        
        if metrics:
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("R² Score", f"{metrics.get('r2', 0):.4f}")
            with metric_cols[1]:
                st.metric("RMSE", format_currency(metrics.get('rmse', 0)))
            with metric_cols[2]:
                st.metric("MAE", format_currency(metrics.get('mae', 0)))
            with metric_cols[3]:
                st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
    
    with col_info2:
        st.markdown("### 📜 Historial")
        display_prediction_history()
    
    # Información sobre el modelo
    with st.expander("ℹ️ Acerca del Modelo"):
        st.markdown("""
        ### Información del Modelo Predictivo
        
        El modelo utilizado para las predicciones es **XGBoost** (Extreme Gradient Boosting), 
        un algoritmo de machine learning basado en árboles de decisión con gradient boosting.
        
        **Características del Modelo:**
        - Entrenado con 1,728 propiedades de Saratoga County, NY (2006)
        - Validación cruzada con 5 folds
        - Optimización de hiperparámetros mediante búsqueda aleatoria
        
        **Limitaciones:**
        - Las predicciones son estimaciones basadas en datos históricos
        - El intervalo de confianza del 95% indica el rango donde probablemente se encuentra el valor real
        - Factores externos no capturados en los datos pueden afectar el precio real
        
        **Variables más importantes:**
        1. Área habitable (livingArea)
        2. Valor del terreno (landValue)
        3. Número de baños (bathrooms)
        4. Número de habitaciones (rooms)
        5. Antigüedad (age)
        """)