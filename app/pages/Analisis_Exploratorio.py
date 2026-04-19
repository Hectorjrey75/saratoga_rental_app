"""
Página de Análisis Exploratorio de Datos.
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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.config.settings import config, RAW_DATA_DIR
from src.data.preprocessing import DataPreprocessor
from src.utils.helpers import format_currency
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Análisis Exploratorio - Rental Price Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Análisis Exploratorio de Datos")
st.markdown("---")


@st.cache_data
def load_data():
    """Cargar datos del dataset."""
    data_path = RAW_DATA_DIR / "SaratogaHouses.csv"
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        logger.info(f"Datos cargados: {len(df)} registros")
        return df
    else:
        st.error(f"❌ No se encontró el archivo de datos en: {data_path}")
        return None


@st.cache_data
def get_numeric_stats(df):
    """Calcular estadísticas numéricas."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    stats_df = df[numeric_cols].describe().T
    stats_df['range'] = stats_df['max'] - stats_df['min']
    stats_df['skewness'] = df[numeric_cols].skew()
    stats_df['kurtosis'] = df[numeric_cols].kurtosis()
    stats_df['cv_pct'] = (stats_df['std'] / stats_df['mean'] * 100)
    
    return stats_df.round(2)


def plot_distribution(df, column, bins=50):
    """Crear histograma con KDE."""
    fig = go.Figure()
    
    # Histograma
    fig.add_trace(go.Histogram(
        x=df[column],
        nbinsx=bins,
        name='Histograma',
        marker_color='#1f77b4',
        opacity=0.7,
        histnorm='probability density'
    ))
    
    # KDE
    data_clean = df[column].dropna()
    if len(data_clean) > 1:
        kde_x = np.linspace(data_clean.min(), data_clean.max(), 100)
        kde = stats.gaussian_kde(data_clean)
        kde_y = kde(kde_x)
        
        fig.add_trace(go.Scatter(
            x=kde_x, y=kde_y,
            mode='lines',
            name='KDE',
            line=dict(color='red', width=2)
        ))
    
    fig.update_layout(
        title=f'Distribución de {column}',
        xaxis_title=column,
        yaxis_title='Densidad',
        showlegend=True,
        height=400
    )
    
    return fig


def plot_boxplot(df, column, groupby=None):
    """Crear boxplot."""
    if groupby:
        fig = px.box(df, x=groupby, y=column, color=groupby, title=f'Boxplot de {column} por {groupby}')
    else:
        fig = px.box(df, y=column, title=f'Boxplot de {column}')
    
    fig.update_layout(height=400)
    return fig


def plot_correlation_heatmap(df):
    """Crear heatmap de correlación."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 9},
        colorbar={"title": "Correlación"}
    ))
    
    fig.update_layout(
        title='Matriz de Correlación - Variables Numéricas',
        height=600,
        width=800
    )
    
    return fig


def plot_scatter_matrix(df, columns, color_by=None):
    """Crear matriz de scatter plots."""
    if color_by and color_by in df.columns:
        fig = px.scatter_matrix(
            df, dimensions=columns,
            color=color_by,
            title='Matriz de Scatter Plots',
            opacity=0.5
        )
    else:
        fig = px.scatter_matrix(
            df, dimensions=columns,
            title='Matriz de Scatter Plots',
            opacity=0.5
        )
    
    fig.update_layout(height=700)
    return fig


def plot_categorical_analysis(df, cat_col, target_col='price'):
    """Análisis de variable categórica vs precio."""
    cat_means = df.groupby(cat_col)[target_col].mean().sort_values(ascending=False)
    cat_counts = df[cat_col].value_counts()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'Precio Promedio por {cat_col}',
            f'Frecuencia de {cat_col}'
        )
    )
    
    # Precio promedio
    fig.add_trace(
        go.Bar(
            x=cat_means.index,
            y=cat_means.values,
            text=[format_currency(v) for v in cat_means.values],
            textposition='outside',
            marker_color='#1f77b4',
            name='Precio Promedio'
        ),
        row=1, col=1
    )
    
    # Frecuencia
    fig.add_trace(
        go.Bar(
            x=cat_counts.index,
            y=cat_counts.values,
            text=cat_counts.values,
            textposition='outside',
            marker_color='#ff7f0e',
            name='Frecuencia'
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig


def plot_outlier_analysis(df, column):
    """Análisis de outliers usando IQR."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    fig = go.Figure()
    
    # Boxplot
    fig.add_trace(go.Box(
        y=df[column],
        name='Boxplot',
        boxmean='sd',
        marker_color='#1f77b4'
    ))
    
    # Outliers
    if len(outliers) > 0:
        fig.add_trace(go.Scatter(
            x=[0] * len(outliers),
            y=outliers[column],
            mode='markers',
            name=f'Outliers ({len(outliers)})',
            marker=dict(color='red', size=8, symbol='x')
        ))
    
    fig.update_layout(
        title=f'Análisis de Outliers - {column}',
        yaxis_title=column,
        height=400,
        showlegend=True
    )
    
    return fig, len(outliers), len(outliers)/len(df)*100


# Cargar datos
df = load_data()

if df is not None:
    # Sidebar para filtros
    with st.sidebar:
        st.header("🎛️ Filtros")
        
        # Filtro por rango de precio
        price_range = st.slider(
            "Rango de Precio",
            min_value=float(df['price'].min()),
            max_value=float(df['price'].max()),
            value=(float(df['price'].min()), float(df['price'].max())),
            step=1000.0
        )
        
        # Filtro por dormitorios
        bedroom_options = sorted(df['bedrooms'].unique())
        selected_bedrooms = st.multiselect(
            "Dormitorios",
            bedroom_options,
            default=bedroom_options[:5]
        )
        
        # Filtro por baños
        bathroom_options = sorted(df['bathrooms'].unique())
        selected_bathrooms = st.multiselect(
            "Baños",
            bathroom_options,
            default=bathroom_options[:5]
        )
        
        # Aplicar filtros
        df_filtered = df[
            (df['price'] >= price_range[0]) & 
            (df['price'] <= price_range[1]) &
            (df['bedrooms'].isin(selected_bedrooms)) &
            (df['bathrooms'].isin(selected_bathrooms))
        ]
        
        st.metric("Registros Filtrados", len(df_filtered))
    
    # Métricas principales
    st.markdown("### 📈 Métricas Generales")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Registros", f"{len(df_filtered):,}")
    with col2:
        st.metric("Precio Promedio", format_currency(df_filtered['price'].mean()))
    with col3:
        st.metric("Precio Mediano", format_currency(df_filtered['price'].median()))
    with col4:
        st.metric("Área Promedio", f"{df_filtered['livingArea'].mean():,.0f} m²")
    with col5:
        st.metric("Dormitorios Prom.", f"{df_filtered['bedrooms'].mean():.1f}")
    with col6:
        st.metric("Antigüedad Prom.", f"{df_filtered['age'].mean():,.0f} años")
    
    st.markdown("---")
    
    # Tabs para diferentes análisis
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Distribuciones",
        "🔗 Correlaciones",
        "📈 Scatter Matrix",
        "🏷️ Categóricas",
        "⚠️ Outliers",
        "📋 Datos"
    ])
    
    with tab1:
        st.markdown("### Distribuciones de Variables")
        
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            selected_num = st.selectbox(
                "Selecciona una variable numérica",
                numeric_cols,
                index=numeric_cols.index('price') if 'price' in numeric_cols else 0
            )
            
            bins = st.slider("Número de bins", 10, 100, 50)
        
        with col2:
            fig = plot_distribution(df_filtered, selected_num, bins)
            st.plotly_chart(fig, use_container_width=True)
            
            # Estadísticas
            stats_dict = {
                "Media": df_filtered[selected_num].mean(),
                "Mediana": df_filtered[selected_num].median(),
                "Desv. Estándar": df_filtered[selected_num].std(),
                "Mínimo": df_filtered[selected_num].min(),
                "Máximo": df_filtered[selected_num].max(),
                "Skewness": df_filtered[selected_num].skew(),
                "Kurtosis": df_filtered[selected_num].kurtosis()
            }
            
            if selected_num == 'price':
                stats_dict = {k: format_currency(v) if isinstance(v, (int, float)) and k in ['Media', 'Mediana', 'Desv. Estándar', 'Mínimo', 'Máximo'] else f"{v:.4f}" if isinstance(v, float) else v for k, v in stats_dict.items()}
            else:
                stats_dict = {k: f"{v:,.2f}" if isinstance(v, float) else v for k, v in stats_dict.items()}
            
            st.json(stats_dict)
        
        # Boxplot
        st.markdown("#### Boxplot")
        groupby_options = ['None'] + df_filtered.select_dtypes(include=['object']).columns.tolist()
        selected_groupby = st.selectbox("Agrupar por (opcional)", groupby_options)
        
        fig_box = plot_boxplot(
            df_filtered, 
            selected_num, 
            selected_groupby if selected_groupby != 'None' else None
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    with tab2:
        st.markdown("### Matriz de Correlación")
        
        # Heatmap de correlación
        fig_corr = plot_correlation_heatmap(df_filtered)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Correlaciones con el precio
        st.markdown("#### Correlaciones con el Precio")
        
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        if 'price' in numeric_cols:
            price_corr = df_filtered[numeric_cols].corr()['price'].drop('price').sort_values(ascending=False)
            
            corr_df = pd.DataFrame({
                'Variable': price_corr.index,
                'Correlación': price_corr.values,
                'Magnitud': ['Fuerte' if abs(c) > 0.5 else 'Moderada' if abs(c) > 0.3 else 'Débil' for c in price_corr.values]
            })
            
            fig = px.bar(
                corr_df, x='Correlación', y='Variable', orientation='h',
                title='Correlación de Variables con el Precio',
                color='Magnitud',
                color_discrete_map={'Fuerte': '#2ecc71', 'Moderada': '#f1c40f', 'Débil': '#e74c3c'},
                text=corr_df['Correlación'].round(3)
            )
            fig.update_traces(textposition='outside')
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Matriz de Scatter Plots")
        
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        
        # Seleccionar variables clave
        default_vars = ['price', 'livingArea', 'landValue', 'age', 'bedrooms', 'bathrooms']
        available_defaults = [v for v in default_vars if v in numeric_cols]
        
        selected_vars = st.multiselect(
            "Selecciona variables para la matriz (máximo 6)",
            numeric_cols,
            default=available_defaults[:6]
        )
        
        if len(selected_vars) > 1:
            color_options = ['None'] + df_filtered.select_dtypes(include=['object']).columns.tolist()
            color_by = st.selectbox("Colorear por", color_options)
            
            fig = plot_scatter_matrix(
                df_filtered.sample(min(1000, len(df_filtered))),
                selected_vars[:6],
                color_by if color_by != 'None' else None
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Selecciona al menos 2 variables para la matriz de scatter plots")
    
    with tab4:
        st.markdown("### Análisis de Variables Categóricas")
        
        cat_cols = df_filtered.select_dtypes(include=['object']).columns.tolist()
        
        if cat_cols:
            selected_cat = st.selectbox("Selecciona una variable categórica", cat_cols)
            
            fig = plot_categorical_analysis(df_filtered, selected_cat)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabla de resumen
            st.markdown("#### Tabla de Resumen")
            summary = df_filtered.groupby(selected_cat).agg({
                'price': ['count', 'mean', 'median', 'std', 'min', 'max']
            }).round(2)
            summary.columns = ['Conteo', 'Media', 'Mediana', 'Desv. Est.', 'Mínimo', 'Máximo']
            st.dataframe(summary)
        else:
            st.info("No hay variables categóricas disponibles")
    
    with tab5:
        st.markdown("### Análisis de Outliers")
        
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            selected_outlier = st.selectbox(
                "Selecciona variable para análisis de outliers",
                numeric_cols
            )
        
        with col2:
            fig, n_outliers, pct_outliers = plot_outlier_analysis(df_filtered, selected_outlier)
            st.plotly_chart(fig, use_container_width=True)
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Outliers Detectados", n_outliers)
            with col_b:
                st.metric("% de Outliers", f"{pct_outliers:.2f}%")
            with col_c:
                Q1 = df_filtered[selected_outlier].quantile(0.25)
                Q3 = df_filtered[selected_outlier].quantile(0.75)
                IQR = Q3 - Q1
                st.metric("IQR", f"{IQR:,.2f}")
        
        # Resumen de outliers para todas las variables
        st.markdown("#### Resumen de Outliers por Variable")
        
        outlier_summary = []
        for col in numeric_cols:
            Q1 = df_filtered[col].quantile(0.25)
            Q3 = df_filtered[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            n_out = len(df_filtered[(df_filtered[col] < lower) | (df_filtered[col] > upper)])
            pct_out = n_out / len(df_filtered) * 100
            
            outlier_summary.append({
                'Variable': col,
                'Outliers': n_out,
                '% Outliers': pct_out,
                'Límite Inferior': lower,
                'Límite Superior': upper
            })
        
        outlier_df = pd.DataFrame(outlier_summary).sort_values('% Outliers', ascending=False)
        st.dataframe(outlier_df.style.format({
            '% Outliers': '{:.2f}%',
            'Límite Inferior': '{:,.2f}',
            'Límite Superior': '{:,.2f}'
        }))
    
    with tab6:
        st.markdown("### Vista de Datos")
        
        # Mostrar datos filtrados
        st.dataframe(df_filtered, use_container_width=True)
        
        # Opción de descarga
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            "📥 Descargar Datos Filtrados (CSV)",
            csv,
            "saratoga_houses_filtered.csv",
            "text/csv"
        )
        
        # Estadísticas descriptivas completas
        with st.expander("📊 Estadísticas Descriptivas Completas"):
            st.dataframe(get_numeric_stats(df_filtered), use_container_width=True)

else:
    st.error("❌ No se pudieron cargar los datos. Verifica que el archivo SaratogaHouses.csv esté en data/raw/")