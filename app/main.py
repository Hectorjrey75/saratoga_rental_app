"""
Main Streamlit application for rental price prediction.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from loguru import logger

from src.config.settings import config, MODELS_DIR, RAW_DATA_DIR
from src.data.preprocessing import DataPreprocessor
from src.models.model_training import ModelTrainer
from src.prediction.prediction import RentalPricePredictor
from src.utils.helpers import format_currency

# Page configuration
st.set_page_config(
    page_title=config.app.title,
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "predictor" not in st.session_state:
    st.session_state.predictor = None
if "preprocessor" not in st.session_state:
    st.session_state.preprocessor = None
if "data" not in st.session_state:
    st.session_state.data = None
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False


def load_model():
    """Load trained model and preprocessor."""
    model_path = MODELS_DIR / "best_model.pkl"
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    
    if model_path.exists() and preprocessor_path.exists():
        try:
            predictor = RentalPricePredictor(model_path, preprocessor_path)
            st.session_state.predictor = predictor
            st.session_state.model_trained = True
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            st.error(f"Error loading model: {e}")
    
    return False


def load_data():
    """Load data from CSV."""
    data_path = RAW_DATA_DIR / "SaratogaHouses.csv"
    
    if data_path.exists():
        try:
            df = pd.read_csv(data_path)
            st.session_state.data = df
            return True
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            st.error(f"Error loading data: {e}")
    
    return False


def train_model():
    """Train a new model."""
    if st.session_state.data is None:
        st.error("No data available for training")
        return False
    
    with st.spinner("Training model... This may take a few minutes."):
        try:
            # Initialize preprocessor
            preprocessor = DataPreprocessor(scaler_type="standard")
            
            # Clean and prepare data
            df_clean = preprocessor.clean_data(st.session_state.data)
            
            # Prepare features
            X = preprocessor.prepare_features(df_clean, fit=True)
            y = df_clean[config.data.target_column]
            
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
            
            # Initialize and train model
            trainer = ModelTrainer(model_type="xgboost")
            
            # Hyperparameter tuning
            trainer.hyperparameter_tuning(
                X_train, y_train,
                search_type="random",
                cv_folds=3
            )
            
            # Evaluate on test set
            metrics = trainer.evaluate(X_test, y_test)
            
            # Cross-validation
            cv_scores = trainer.cross_validate(X, y, cv_folds=5)
            
            # Save model and preprocessor
            trainer.save_model(MODELS_DIR / "best_model.pkl")
            preprocessor.save_preprocessor(MODELS_DIR / "preprocessor.pkl")
            
            # Save processed data
            preprocessor.save_processed_data(
                X_train, X_val, X_test, y_train, y_val, y_test,
                config.PROCESSED_DATA_DIR
            )
            
            # Update session state
            predictor = RentalPricePredictor(
                MODELS_DIR / "best_model.pkl",
                MODELS_DIR / "preprocessor.pkl"
            )
            st.session_state.predictor = predictor
            st.session_state.preprocessor = preprocessor
            st.session_state.model_trained = True
            
            st.success("Model trained successfully!")
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R² Score", f"{metrics['r2']:.4f}")
            with col2:
                st.metric("RMSE", format_currency(metrics['rmse']))
            with col3:
                st.metric("MAE", format_currency(metrics['mae']))
            with col4:
                st.metric("MAPE", f"{metrics['mape']:.2f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            st.error(f"Training failed: {e}")
            return False


def sidebar():
    """Render sidebar."""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/house.png", width=80)
        st.title("🏠 Rental Price Predictor")
        st.markdown("---")
        
        # Navigation
        selected = option_menu(
            menu_title=None,
            options=["Home", "Data Analysis", "Prediction", "Batch Prediction", "Scenarios", "Model Info"],
            icons=["house", "graph-up", "calculator", "table", "gear", "info-circle"],
            default_index=0,
        )
        
        st.markdown("---")
        
        # Model status
        if st.session_state.model_trained:
            st.success("✅ Model Loaded")
            if st.button("🔄 Reload Model"):
                load_model()
                st.rerun()
        else:
            st.warning("⚠️ No Model Loaded")
            
            # Load data
            if st.button("📂 Load Data"):
                if load_data():
                    st.success(f"Loaded {len(st.session_state.data)} records")
            
            # Train model
            if st.session_state.data is not None:
                if st.button("🚀 Train Model", type="primary"):
                    train_model()
            else:
                st.info("Load data to train model")
        
        st.markdown("---")
        st.caption(f"v{config.app.version}")
        
        return selected


def home_page():
    """Home page content."""
    st.title("🏠 Sistema Profesional de Predicción de Precios de Alquiler")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Bienvenido al Sistema de Predicción de Precios de Alquiler
        
        Este sistema utiliza algoritmos avanzados de Machine Learning para predecir 
        el precio de alquiler de viviendas basado en sus características.
        
        #### 🎯 Funcionalidades Principales:
        
        - **Predicción Individual**: Obtén una estimación precisa para una propiedad específica
        - **Predicción por Lote**: Procesa múltiples propiedades simultáneamente
        - **Análisis de Escenarios**: Compara diferentes configuraciones de propiedades
        - **Análisis Exploratorio**: Visualiza patrones y tendencias en los datos
        - **Forecast Recursivo**: Proyecciones a futuro con intervalos de confianza
        
        #### 📊 Datos del Modelo:
        
        El modelo ha sido entrenado con datos de 1,728 viviendas en Saratoga County, 
        New York (2006), considerando variables como:
        - Tamaño del terreno y área habitable
        - Número de habitaciones y baños
        - Antigüedad de la propiedad
        - Características del vecindario
        - Amenidades (chimeneas, aire acondicionado, etc.)
        """)
    
    with col2:
        st.markdown("### 🏠 Vista Rápida")
        
        if st.session_state.data is not None:
            st.metric("Total de Propiedades", f"{len(st.session_state.data):,}")
            st.metric("Precio Promedio", format_currency(st.session_state.data['price'].mean()))
            
            # Price distribution
            fig = px.histogram(
                st.session_state.data, x="price",
                nbins=50, title="Distribución de Precios",
                labels={"price": "Precio ($)"}
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Quick actions
    st.markdown("---")
    st.markdown("### ⚡ Acciones Rápidas")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📂 Cargar Datos", use_container_width=True):
            if load_data():
                st.success("Datos cargados correctamente!")
    
    with col2:
        if st.button("🚀 Entrenar Modelo", use_container_width=True, disabled=st.session_state.data is None):
            train_model()
    
    with col3:
        if st.button("🔮 Nueva Predicción", use_container_width=True, disabled=not st.session_state.model_trained):
            st.session_state.nav_to = "Prediction"
    
    with col4:
        if st.button("📊 Ver Análisis", use_container_width=True, disabled=st.session_state.data is None):
            st.session_state.nav_to = "Data Analysis"


def prediction_page():
    """Individual prediction page."""
    st.title("🔮 Predicción Individual de Precio")
    st.markdown("---")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ No hay modelo cargado. Por favor, entrena o carga un modelo primero.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Características de la Propiedad")
        
        with st.form("prediction_form"):
            # Property characteristics
            lot_size = st.number_input("Tamaño del Terreno (m²)", min_value=0.0, value=0.5, step=0.1)
            age = st.number_input("Antigüedad (años)", min_value=0, value=15, step=1)
            land_value = st.number_input("Valor del Terreno ($)", min_value=0, value=50000, step=1000)
            living_area = st.number_input("Área Habitable (m²)", min_value=0, value=1500, step=50)
            pct_college = st.slider("Porcentaje Universitario (%)", 0, 100, 50)
            
            col_a, col_b = st.columns(2)
            with col_a:
                bedrooms = st.number_input("Dormitorios", min_value=1, value=3, step=1)
                fireplaces = st.number_input("Chimeneas", min_value=0, value=1, step=1)
            with col_b:
                bathrooms = st.number_input("Baños", min_value=0.5, value=2.0, step=0.5)
                rooms = st.number_input("Habitaciones", min_value=1, value=6, step=1)
            
            # Categorical features
            heating = st.selectbox("Tipo de Calefacción", ["hot air", "hot water/steam", "electric"])
            fuel = st.selectbox("Combustible", ["gas", "oil", "electric"])
            sewer = st.selectbox("Desagüe", ["septic", "public/commercial", "none"])
            waterfront = st.selectbox("Vista al Lago", ["No", "Yes"])
            new_construction = st.selectbox("Nueva Construcción", ["No", "Yes"])
            central_air = st.selectbox("Aire Acondicionado", ["No", "Yes"])
            
            submitted = st.form_submit_button("🔮 Predecir Precio", type="primary")
    
    with col2:
        st.markdown("### 📊 Resultado de la Predicción")
        
        if submitted:
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
            
            with st.spinner("Calculando predicción..."):
                result = st.session_state.predictor.predict_single(features)
            
            # Display result
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 30px; border-radius: 15px; text-align: center;">
                <h3 style="color: white; margin-bottom: 10px;">Precio Estimado de Alquiler</h3>
                <h1 style="color: white; font-size: 48px;">{result['predicted_price_formatted']}</h1>
                <p style="color: rgba(255,255,255,0.9);">
                    Intervalo de Confianza (95%): 
                    {result['confidence_interval']['lower_formatted']} - 
                    {result['confidence_interval']['upper_formatted']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature impact
            st.markdown("### 📈 Impacto de Características")
            
            if st.session_state.predictor.get_feature_importance() is not None:
                importance_df = st.session_state.predictor.get_feature_importance().head(10)
                
                fig = px.bar(
                    importance_df, x="importance", y="feature",
                    orientation="h", title="Top 10 Características Más Influyentes",
                    labels={"importance": "Importancia", "feature": "Característica"}
                )
                st.plotly_chart(fig, use_container_width=True)


def batch_prediction_page():
    """Batch prediction page."""
    st.title("📊 Predicción por Lote")
    st.markdown("---")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ No hay modelo cargado. Por favor, entrena o carga un modelo primero.")
        return
    
    st.markdown("### 📤 Cargar Archivo de Propiedades")
    
    uploaded_file = st.file_uploader(
        "Selecciona un archivo CSV con las propiedades a predecir",
        type=["csv"]
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Archivo cargado: {len(df)} propiedades")
            
            st.markdown("### 📋 Vista Previa de Datos")
            st.dataframe(df.head(10))
            
            if st.button("🚀 Realizar Predicciones", type="primary"):
                with st.spinner("Procesando predicciones..."):
                    results = st.session_state.predictor.predict_batch(df)
                
                st.markdown("### 📊 Resultados de Predicciones")
                
                # Display results
                st.dataframe(results)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Precio Promedio", 
                             format_currency(results['predicted_price'].mean()))
                with col2:
                    st.metric("Precio Mínimo", 
                             format_currency(results['predicted_price'].min()))
                with col3:
                    st.metric("Precio Máximo", 
                             format_currency(results['predicted_price'].max()))
                
                # Distribution plot
                fig = px.histogram(
                    results, x="predicted_price",
                    nbins=30, title="Distribución de Precios Predichos",
                    labels={"predicted_price": "Precio Predicho ($)"}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Download button
                csv = results.to_csv(index=False)
                st.download_button(
                    "📥 Descargar Resultados (CSV)",
                    csv,
                    f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
                
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")


def scenarios_page():
    """Scenario analysis page."""
    st.title("🎯 Análisis de Escenarios")
    st.markdown("---")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ No hay modelo cargado. Por favor, entrena o carga un modelo primero.")
        return
    
    st.markdown("### 🔧 Configuración Base")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("Características Base", expanded=True):
            # Base features
            base_lot = st.number_input("Tamaño Terreno (m²)", value=0.5, key="base_lot")
            base_age = st.number_input("Antigüedad", value=15, key="base_age")
            base_land = st.number_input("Valor Terreno ($)", value=50000, key="base_land")
            base_living = st.number_input("Área Habitable", value=1500, key="base_living")
            base_college = st.slider("% Universitario", 0, 100, 50, key="base_college")
            base_bedrooms = st.number_input("Dormitorios", value=3, key="base_bed")
            base_bathrooms = st.number_input("Baños", value=2.0, key="base_bath")
    
    with col2:
        with st.expander("Escenarios a Comparar", expanded=True):
            scenarios = st.multiselect(
                "Selecciona escenarios",
                ["Mejora de Eficiencia", "Añadir Amenidades", "Renovación", "Expansión"],
                default=["Mejora de Eficiencia"]
            )
    
    if st.button("📊 Analizar Escenarios", type="primary"):
        base_features = {
            "lotSize": base_lot, "age": base_age, "landValue": base_land,
            "livingArea": base_living, "pctCollege": base_college,
            "bedrooms": base_bedrooms, "bathrooms": base_bathrooms,
            "fireplaces": 1, "rooms": base_bedrooms + 2,
            "heating": "hot air", "fuel": "gas", "sewer": "septic",
            "waterfront": "No", "newConstruction": "No", "centralAir": "No"
        }
        
        # Define scenarios
        scenario_configs = {
            "Mejora de Eficiencia": {
                "name": "Mejora de Eficiencia",
                "description": "Añadir aire acondicionado central",
                "modifications": {"centralAir": "Yes"}
            },
            "Añadir Amenidades": {
                "name": "Añadir Amenidades",
                "description": "Añadir chimenea",
                "modifications": {"fireplaces": 2}
            },
            "Renovación": {
                "name": "Renovación",
                "description": "Reducir antigüedad en 10 años",
                "modifications": {"age": max(0, base_age - 10)}
            },
            "Expansión": {
                "name": "Expansión",
                "description": "Añadir 200 m² de área habitable",
                "modifications": {"livingArea": base_living + 200}
            }
        }
        
        selected_scenarios = [scenario_configs[s] for s in scenarios]
        
        with st.spinner("Analizando escenarios..."):
            results = st.session_state.predictor.analyze_scenario(
                base_features, selected_scenarios
            )
        
        st.markdown("### 📊 Resultados del Análisis")
        
        # Display results table
        st.dataframe(results.style.format({
            "predicted_price": "${:,.2f}",
            "price_change": "${:,.2f}",
            "price_change_pct": "{:.1f}%"
        }))
        
        # Bar chart comparison
        fig = px.bar(
            results, x="scenario", y="predicted_price",
            title="Comparación de Escenarios",
            labels={"predicted_price": "Precio Predicho ($)", "scenario": "Escenario"},
            color="scenario",
            text=results["predicted_price"].apply(lambda x: format_currency(x))
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main application entry point."""
    # Sidebar navigation
    selected = sidebar()
    
    # Page routing
    if selected == "Home":
        home_page()
    elif selected == "Data Analysis":
        st.title("📈 Análisis Exploratorio de Datos")
        st.markdown("---")
        
        if st.session_state.data is not None:
            df = st.session_state.data
            
            # Overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Registros", len(df))
            with col2:
                st.metric("Variables", len(df.columns))
            with col3:
                st.metric("Precio Promedio", format_currency(df['price'].mean()))
            with col4:
                st.metric("Precio Mediano", format_currency(df['price'].median()))
            
            # Tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs(["Distribuciones", "Correlaciones", "Categóricas", "Datos"])
            
            with tab1:
                st.subheader("Distribución de Variables Numéricas")
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                selected_col = st.selectbox("Selecciona variable", num_cols)
                
                fig = px.histogram(
                    df, x=selected_col,
                    nbins=50, title=f"Distribución de {selected_col}",
                    marginal="box"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("Matriz de Correlación")
                num_df = df.select_dtypes(include=[np.number])
                corr_matrix = num_df.corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    aspect="auto",
                    title="Correlación entre Variables Numéricas",
                    color_continuous_scale="RdBu_r"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("Análisis de Variables Categóricas")
                cat_cols = df.select_dtypes(include=['object']).columns.tolist()
                selected_cat = st.selectbox("Selecciona variable categórica", cat_cols)
                
                fig = px.bar(
                    df[selected_cat].value_counts().reset_index(),
                    x=selected_cat, y="count",
                    title=f"Frecuencia de {selected_cat}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.subheader("Vista de Datos")
                st.dataframe(df)
        else:
            st.info("No hay datos cargados. Carga los datos desde la barra lateral.")
    
    elif selected == "Prediction":
        prediction_page()
    
    elif selected == "Batch Prediction":
        batch_prediction_page()
    
    elif selected == "Scenarios":
        scenarios_page()
    
    elif selected == "Model Info":
        st.title("ℹ️ Información del Modelo")
        st.markdown("---")
        
        if st.session_state.model_trained:
            metrics = st.session_state.predictor.get_model_metrics()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{metrics.get('r2', 0):.4f}")
                st.metric("MAE", format_currency(metrics.get('mae', 0)))
            with col2:
                st.metric("RMSE", format_currency(metrics.get('rmse', 0)))
                st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
            with col3:
                st.metric("Max Error", format_currency(metrics.get('max_error', 0)))
                st.metric("Med. Abs. Error", format_currency(metrics.get('median_absolute_error', 0)))
            
            # Feature importance
            st.markdown("### 📊 Importancia de Características")
            importance_df = st.session_state.predictor.get_feature_importance()
            
            if importance_df is not None:
                fig = px.bar(
                    importance_df.head(15), x="importance", y="feature",
                    orientation="h", title="Top 15 Características Más Influyentes",
                    labels={"importance": "Importancia", "feature": "Característica"}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay modelo cargado. Entrena o carga un modelo primero.")


if __name__ == "__main__":
    # Auto-load model on startup
    load_model()
    main()