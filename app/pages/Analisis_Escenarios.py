"""
Página de Análisis de Escenarios.
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

from src.config.settings import config, MODELS_DIR
from src.prediction.prediction import RentalPricePredictor
from src.utils.helpers import format_currency
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Análisis de Escenarios - Rental Price Predictor",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Análisis de Escenarios")
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


def display_scenario_comparison(results_df):
    """Mostrar comparación de escenarios."""
    
    # Métricas de comparación
    st.markdown("### 📊 Comparación de Escenarios")
    
    # Formatear tabla
    display_df = results_df.copy()
    display_df['Precio Predicho'] = display_df['predicted_price'].apply(format_currency)
    display_df['Cambio'] = display_df['price_change'].apply(lambda x: f"+{format_currency(x)}" if x > 0 else format_currency(x))
    display_df['Cambio %'] = display_df['price_change_pct'].apply(lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%")
    
    # Tabla estilizada
    st.dataframe(
        display_df[['scenario', 'description', 'Precio Predicho', 'Cambio', 'Cambio %']],
        use_container_width=True,
        hide_index=True
    )
    
    # Gráfico de barras comparativo
    st.markdown("### 📈 Comparación Visual")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        fig_bar = px.bar(
            results_df, x='scenario', y='predicted_price',
            title='Precio por Escenario',
            labels={'predicted_price': 'Precio Predicho ($)', 'scenario': 'Escenario'},
            color='scenario',
            text=results_df['predicted_price'].apply(lambda x: format_currency(x)),
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col_chart2:
        # Gráfico de cascada (waterfall) para mostrar cambios
        base_price = results_df[results_df['scenario'] == 'Base']['predicted_price'].values[0]
        
        waterfall_data = []
        running_total = base_price
        
        for _, row in results_df.iterrows():
            if row['scenario'] != 'Base':
                waterfall_data.append({
                    'measure': 'relative',
                    'label': row['scenario'],
                    'value': row['price_change']
                })
                running_total += row['price_change']
        
        waterfall_data.append({
            'measure': 'total',
            'label': 'Total',
            'value': running_total
        })
        
        fig_waterfall = go.Figure(go.Waterfall(
            name="Cambios",
            orientation="v",
            measure=[d['measure'] for d in waterfall_data],
            x=[d['label'] for d in waterfall_data],
            y=[d['value'] for d in waterfall_data],
            text=[format_currency(d['value']) for d in waterfall_data],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#e74c3c"}},
            increasing={"marker": {"color": "#2ecc71"}},
            totals={"marker": {"color": "#3498db"}}
        ))
        
        fig_waterfall.update_layout(
            title="Análisis de Cascada - Cambios en el Precio",
            height=450,
            showlegend=False
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)
    
    # ROI potencial
    st.markdown("### 💰 Análisis de Retorno de Inversión (ROI)")
    
    roi_data = []
    base_price = results_df[results_df['scenario'] == 'Base']['predicted_price'].values[0]
    
    # Estimaciones de costo para mejoras
    cost_estimates = {
        "Mejora de Eficiencia": 5000,
        "Añadir Amenidades": 3000,
        "Renovación": 15000,
        "Expansión": 30000,
        "Vista al Lago": 0,  # No se puede comprar
        "Nueva Construcción": 50000,
        "Aire Acondicionado": 5000,
        "Chimenea Adicional": 2000
    }
    
    for _, row in results_df.iterrows():
        if row['scenario'] != 'Base':
            scenario_name = row['scenario']
            estimated_cost = cost_estimates.get(scenario_name, 5000)
            value_increase = row['price_change']
            roi = (value_increase / estimated_cost * 100) if estimated_cost > 0 else float('inf')
            payback_years = estimated_cost / (value_increase * 0.05) if value_increase > 0 else float('inf')  # Asumiendo 5% de retorno anual
            
            roi_data.append({
                'Escenario': scenario_name,
                'Inversión Estimada': estimated_cost,
                'Incremento de Valor': value_increase,
                'ROI (%)': roi,
                'Payback (años)': payback_years
            })
    
    if roi_data:
        roi_df = pd.DataFrame(roi_data)
        roi_df['Inversión Estimada'] = roi_df['Inversión Estimada'].apply(format_currency)
        roi_df['Incremento de Valor'] = roi_df['Incremento de Valor'].apply(format_currency)
        roi_df['ROI (%)'] = roi_df['ROI (%)'].apply(lambda x: f"{x:.1f}%" if x != float('inf') else "∞")
        roi_df['Payback (años)'] = roi_df['Payback (años)'].apply(lambda x: f"{x:.1f}" if x != float('inf') else "∞")
        
        st.dataframe(roi_df, use_container_width=True, hide_index=True)


def display_sensitivity_analysis(predictor, base_features):
    """Mostrar análisis de sensibilidad."""
    st.markdown("### 📊 Análisis de Sensibilidad")
    
    st.markdown("""
    El análisis de sensibilidad muestra cómo cambia el precio predicho al variar 
    una característica específica mientras se mantienen las demás constantes.
    """)
    
    # Seleccionar característica para análisis
    numeric_features = ['livingArea', 'lotSize', 'age', 'landValue', 'pctCollege', 'bedrooms', 'bathrooms', 'rooms']
    
    selected_feature = st.selectbox(
        "Selecciona una característica para analizar su sensibilidad",
        numeric_features
    )
    
    if selected_feature:
        # Definir rango de valores
        base_value = base_features[selected_feature]
        
        if selected_feature in ['bedrooms', 'rooms']:
            values = list(range(max(1, int(base_value) - 2), int(base_value) + 4))
        elif selected_feature == 'bathrooms':
            values = np.arange(max(0.5, base_value - 1.5), base_value + 2.0, 0.5).tolist()
        elif selected_feature == 'age':
            values = list(range(max(0, int(base_value) - 20), int(base_value) + 30, 5))
        else:
            pct_range = 0.5
            values = np.linspace(base_value * (1 - pct_range), base_value * (1 + pct_range), 10).tolist()
        
        with st.spinner("Calculando análisis de sensibilidad..."):
            try:
                sensitivity_results = predictor.sensitivity_analysis(
                    base_features, selected_feature, values
                )
                
                # Gráfico de sensibilidad
                fig_sens = px.line(
                    sensitivity_results, x=selected_feature, y='predicted_price',
                    title=f'Sensibilidad del Precio a {selected_feature}',
                    labels={selected_feature: selected_feature, 'predicted_price': 'Precio Predicho ($)'},
                    markers=True
                )
                
                # Añadir banda de confianza
                fig_sens.add_trace(
                    go.Scatter(
                        x=sensitivity_results[selected_feature].tolist() + sensitivity_results[selected_feature].tolist()[::-1],
                        y=sensitivity_results['ci_upper'].tolist() + sensitivity_results['ci_lower'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(31, 119, 180, 0.2)',
                        line=dict(color='rgba(255, 255, 255, 0)'),
                        hoverinfo="skip",
                        showlegend=True,
                        name='Intervalo 95%'
                    )
                )
                
                fig_sens.update_layout(height=450)
                st.plotly_chart(fig_sens, use_container_width=True)
                
                # Efecto marginal
                if 'marginal_effect' in sensitivity_results.columns:
                    st.markdown("#### Efecto Marginal")
                    
                    fig_marg = px.bar(
                        sensitivity_results.dropna(), x=selected_feature, y='marginal_effect',
                        title=f'Efecto Marginal de {selected_feature}',
                        labels={selected_feature: selected_feature, 'marginal_effect': 'Cambio en Precio ($)'},
                        color='marginal_effect',
                        color_continuous_scale='RdBu'
                    )
                    fig_marg.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig_marg, use_container_width=True)
                
                # Tabla de resultados
                st.markdown("#### Datos del Análisis")
                display_sens = sensitivity_results.copy()
                display_sens['Precio'] = display_sens['predicted_price'].apply(format_currency)
                display_sens['Límite Inf.'] = display_sens['ci_lower'].apply(format_currency)
                display_sens['Límite Sup.'] = display_sens['ci_upper'].apply(format_currency)
                
                st.dataframe(
                    display_sens[[selected_feature, 'Precio', 'Límite Inf.', 'Límite Sup.']],
                    use_container_width=True,
                    hide_index=True
                )
                
            except Exception as e:
                st.error(f"Error en análisis de sensibilidad: {e}")


def display_recursive_forecast(predictor, base_features):
    """Mostrar forecast recursivo."""
    st.markdown("### 📈 Forecast Recursivo Multietapa")
    
    st.markdown("""
    El forecast recursivo proyecta el valor de la propiedad a futuro, considerando 
    una tasa de crecimiento anual y la incertidumbre acumulada en el tiempo.
    """)
    
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        periods = st.slider("Períodos a proyectar (meses)", 1, 60, 24)
    
    with col_f2:
        growth_rate = st.slider("Tasa de crecimiento anual (%)", 0.0, 10.0, 3.0, 0.5) / 100
    
    with st.spinner("Generando forecast..."):
        try:
            forecast_df = predictor.recursive_forecast(base_features, periods, growth_rate)
            
            # Gráfico de forecast
            fig_forecast = go.Figure()
            
            # Línea principal
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['period'], y=forecast_df['forecasted_price'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#1f77b4', width=3)
            ))
            
            # Banda de confianza
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['period'].tolist() + forecast_df['period'].tolist()[::-1],
                y=forecast_df['ci_upper'].tolist() + forecast_df['ci_lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(31, 119, 180, 0.2)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                hoverinfo="skip",
                showlegend=True,
                name='Intervalo 95%'
            ))
            
            fig_forecast.update_layout(
                title='Proyección de Valor de la Propiedad',
                xaxis_title='Meses',
                yaxis_title='Valor Estimado ($)',
                height=450,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Métricas de forecast
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            initial_price = forecast_df[forecast_df['period'] == 0]['forecasted_price'].values[0]
            final_price = forecast_df[forecast_df['period'] == periods]['forecasted_price'].values[0]
            total_growth = final_price - initial_price
            growth_pct = (final_price / initial_price - 1) * 100
            
            with col_m1:
                st.metric("Valor Inicial", format_currency(initial_price))
            with col_m2:
                st.metric("Valor Final", format_currency(final_price))
            with col_m3:
                st.metric("Crecimiento Total", format_currency(total_growth))
            with col_m4:
                st.metric("Crecimiento %", f"{growth_pct:.2f}%")
            
            # Tabla de datos
            with st.expander("📊 Ver Datos del Forecast"):
                display_fc = forecast_df.copy()
                display_fc['Precio'] = display_fc['forecasted_price'].apply(format_currency)
                display_fc['Límite Inf.'] = display_fc['ci_lower'].apply(format_currency)
                display_fc['Límite Sup.'] = display_fc['ci_upper'].apply(format_currency)
                st.dataframe(
                    display_fc[['period', 'Precio', 'Límite Inf.', 'Límite Sup.']],
                    use_container_width=True,
                    hide_index=True
                )
            
        except Exception as e:
            st.error(f"Error en forecast: {e}")


# Cargar predictor
predictor = load_predictor()

if predictor is None:
    st.warning("⚠️ No hay modelo entrenado. Por favor, entrena el modelo desde la página principal.")
else:
    # Configuración de la propiedad base
    st.markdown("### 🏠 Configuración de la Propiedad Base")
    
    with st.expander("📝 Características de la Propiedad Base", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            base_lot = st.number_input("Tamaño Terreno", 0.0, 20.0, 0.5, 0.1, key="base_lot")
            base_age = st.number_input("Antigüedad", 0, 300, 15, 1, key="base_age")
            base_land = st.number_input("Valor Terreno ($)", 0, 500000, 50000, 1000, key="base_land")
            base_living = st.number_input("Área Habitable (m²)", 100, 10000, 1500, 50, key="base_living")
            base_college = st.slider("% Universitario", 0, 100, 50, 1, key="base_college")
        
        with col2:
            base_bed = st.number_input("Dormitorios", 0, 10, 3, 1, key="base_bed")
            base_bath = st.number_input("Baños", 0.0, 8.0, 2.0, 0.5, key="base_bath")
            base_fire = st.number_input("Chimeneas", 0, 5, 1, 1, key="base_fire")
            base_rooms = st.number_input("Habitaciones", 1, 20, 6, 1, key="base_rooms")
        
        with col3:
            base_heat = st.selectbox("Calefacción", ["hot air", "hot water/steam", "electric"], key="base_heat")
            base_fuel = st.selectbox("Combustible", ["gas", "oil", "electric"], key="base_fuel")
            base_sewer = st.selectbox("Desagüe", ["septic", "public/commercial", "none"], key="base_sewer")
            base_water = st.selectbox("Vista al Lago", ["No", "Yes"], key="base_water")
            base_new = st.selectbox("Nueva Construcción", ["No", "Yes"], key="base_new")
            base_air = st.selectbox("Aire Acondicionado", ["No", "Yes"], key="base_air")
    
    # Construir features base
    base_features = {
        "lotSize": base_lot, "age": base_age, "landValue": base_land,
        "livingArea": base_living, "pctCollege": base_college,
        "bedrooms": base_bed, "fireplaces": base_fire,
        "bathrooms": base_bath, "rooms": base_rooms,
        "heating": base_heat, "fuel": base_fuel, "sewer": base_sewer,
        "waterfront": base_water, "newConstruction": base_new,
        "centralAir": base_air
    }
    
    # Tabs para diferentes análisis
    tab1, tab2, tab3 = st.tabs([
        "🎯 Comparación de Escenarios",
        "📊 Análisis de Sensibilidad",
        "📈 Forecast Recursivo"
    ])
    
    with tab1:
        st.markdown("### 🎯 Escenarios a Comparar")
        
        # Escenarios predefinidos
        scenario_options = {
            "Mejora de Eficiencia": {
                "name": "Mejora de Eficiencia",
                "description": "Añadir aire acondicionado central",
                "modifications": {"centralAir": "Yes"}
            },
            "Añadir Amenidades": {
                "name": "Añadir Amenidades",
                "description": "Añadir una chimenea adicional",
                "modifications": {"fireplaces": base_fire + 1}
            },
            "Renovación": {
                "name": "Renovación",
                "description": "Reducir antigüedad en 10 años",
                "modifications": {"age": max(0, base_age - 10)}
            },
            "Expansión": {
                "name": "Expansión",
                "description": "Añadir 300 m² de área habitable",
                "modifications": {"livingArea": base_living + 300}
            },
            "Vista al Lago": {
                "name": "Vista al Lago",
                "description": "Propiedad con vista al lago",
                "modifications": {"waterfront": "Yes"}
            },
            "Nueva Construcción": {
                "name": "Nueva Construcción",
                "description": "Propiedad nueva",
                "modifications": {"newConstruction": "Yes", "age": 0}
            }
        }
        
        selected_scenarios = st.multiselect(
            "Selecciona escenarios a comparar",
            list(scenario_options.keys()),
            default=["Mejora de Eficiencia", "Renovación"]
        )
        
        # Escenario personalizado
        st.markdown("#### 🔧 Escenario Personalizado")
        custom_scenario = st.checkbox("Añadir escenario personalizado")
        
        custom_scenarios = []
        if custom_scenario:
            custom_name = st.text_input("Nombre del escenario", "Personalizado")
            custom_desc = st.text_input("Descripción", "Configuración personalizada")
            
            st.markdown("Modificaciones:")
            col_c1, col_c2, col_c3 = st.columns(3)
            
            with col_c1:
                custom_living = st.number_input("Área Habitable", 100, 10000, base_living, 50, key="custom_living")
                custom_age = st.number_input("Antigüedad", 0, 300, base_age, 1, key="custom_age")
            
            with col_c2:
                custom_bed = st.number_input("Dormitorios", 0, 10, base_bed, 1, key="custom_bed")
                custom_bath = st.number_input("Baños", 0.0, 8.0, base_bath, 0.5, key="custom_bath")
            
            with col_c3:
                custom_water = st.selectbox("Vista al Lago", ["No", "Yes"], key="custom_water")
                custom_air = st.selectbox("Aire Acondicionado", ["No", "Yes"], key="custom_air")
            
            custom_scenarios = [{
                "name": custom_name,
                "description": custom_desc,
                "modifications": {
                    "livingArea": custom_living,
                    "age": custom_age,
                    "bedrooms": custom_bed,
                    "bathrooms": custom_bath,
                    "waterfront": custom_water,
                    "centralAir": custom_air
                }
            }]
        
        if st.button("📊 Analizar Escenarios", type="primary"):
            # Construir lista de escenarios
            scenarios_to_analyze = [scenario_options[s] for s in selected_scenarios] + custom_scenarios
            
            if scenarios_to_analyze:
                with st.spinner("Analizando escenarios..."):
                    try:
                        results = predictor.analyze_scenario(base_features, scenarios_to_analyze)
                        display_scenario_comparison(results)
                        
                    except Exception as e:
                        st.error(f"Error en análisis de escenarios: {e}")
            else:
                st.warning("Selecciona al menos un escenario para analizar")
    
    with tab2:
        display_sensitivity_analysis(predictor, base_features)
    
    with tab3:
        display_recursive_forecast(predictor, base_features)
    
    # Información adicional
    with st.expander("ℹ️ Acerca del Análisis de Escenarios"):
        st.markdown("""
        ### 🎯 Análisis de Escenarios
        
        Esta herramienta permite evaluar cómo diferentes modificaciones a una propiedad 
        afectan su valor estimado.
        
        **Tipos de análisis disponibles:**
        
        1. **Comparación de Escenarios**: Evalúa el impacto de diferentes mejoras o cambios
        
        2. **Análisis de Sensibilidad**: Muestra cómo varía el precio al modificar una sola característica
        
        3. **Forecast Recursivo**: Proyecta el valor futuro considerando apreciación
        
        **Interpretación de resultados:**
        - **Cambio**: Diferencia en dólares respecto al escenario base
        - **Cambio %**: Porcentaje de cambio respecto al valor base
        - **ROI**: Retorno sobre la inversión estimada
        - **Payback**: Años para recuperar la inversión (asumiendo 5% de retorno anual)
        
        **Limitaciones:**
        - Los costos de mejoras son estimaciones generales
        - El forecast asume condiciones de mercado estables
        - Factores externos no modelados pueden afectar los valores reales
        """)