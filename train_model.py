"""
Script para entrenar el modelo de predicción de precios.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from loguru import logger

#from src.config.settings import config, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
#from src.data.preprocessing import DataPreprocessor
#from src.models.model_training import ModelTrainer
#from src.utils.helpers import format_currency


# Importar tanto config como las rutas globales
from src.config.settings import (
    config, 
    RAW_DATA_DIR, 
    PROCESSED_DATA_DIR, 
    MODELS_DIR
)
from src.data.preprocessing import DataPreprocessor
from src.models.model_training import ModelTrainer
from src.utils.helpers import format_currency

def main():
    """Función principal de entrenamiento."""
    logger.info("=" * 80)
    logger.info("🚀 INICIANDO ENTRENAMIENTO DEL MODELO")
    logger.info("=" * 80)
    
    # 1. Cargar datos
    logger.info("📂 Cargando datos...")
    data_path = RAW_DATA_DIR / "SaratogaHouses.csv"
    
    if not data_path.exists():
        logger.error(f"❌ No se encontró el archivo de datos en: {data_path}")
        logger.info(f"   Buscando en: {RAW_DATA_DIR}")
        return
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"✅ Datos cargados: {len(df):,} filas × {len(df.columns)} columnas")
    except Exception as e:
        logger.error(f"❌ Error al cargar datos: {e}")
        return
    
    # 2. Preprocesar datos
    logger.info("🔧 Preprocesando datos...")
    preprocessor = DataPreprocessor(scaler_type="robust")
    
    try:
        # Limpiar datos
        df_clean = preprocessor.clean_data(df)
        logger.info(f"✅ Datos limpiados: {len(df_clean):,} filas")
        
        # Preparar features
        X = preprocessor.prepare_features(df_clean, fit=True)
        y = df_clean[config.data.target_column]
        logger.info(f"✅ Features preparadas: {X.shape[1]} features")
    except Exception as e:
        logger.error(f"❌ Error en preprocesamiento: {e}")
        return
    
    # 3. Dividir datos
    logger.info("📊 Dividiendo datos...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
        
        logger.info(f"   Train: {len(X_train):,} muestras")
        logger.info(f"   Validation: {len(X_val):,} muestras")
        logger.info(f"   Test: {len(X_test):,} muestras")
    except Exception as e:
        logger.error(f"❌ Error al dividir datos: {e}")
        return
    
    # 4. Entrenar modelo
    logger.info("🤖 Entrenando modelo...")
    
    # Intentar con XGBoost primero, si no está disponible usar Random Forest
    try:
        import xgboost
        model_type = "xgboost"
        logger.info("   Usando XGBoost")
    except ImportError:
        model_type = "random_forest"
        logger.warning("   XGBoost no disponible, usando Random Forest")
    
    trainer = ModelTrainer(model_type=model_type)
    
    # Hyperparameter tuning (rápido)
    logger.info("🎯 Optimizando hiperparámetros...")
    try:
        best_params = trainer.hyperparameter_tuning(
            X_train, y_train,
            search_type="random",
            cv_folds=3
        )
        logger.info(f"✅ Mejores parámetros: {best_params}")
    except Exception as e:
        logger.warning(f"⚠️ Hyperparameter tuning falló: {e}")
        logger.info("   Continuando con parámetros por defecto")
        best_params = {}
    
    # Entrenar con mejores parámetros
    logger.info("📈 Entrenando modelo final...")
    try:
        trainer.train(X_train, y_train, X_val, y_val, params=best_params)
    except Exception as e:
        logger.error(f"❌ Error al entrenar modelo: {e}")
        return
    
    # 5. Evaluar modelo
    logger.info("📊 Evaluando modelo...")
    try:
        metrics = trainer.evaluate(X_test, y_test)
        
        logger.info("✅ Métricas en test:")
        logger.info(f"   R² Score: {metrics['r2']:.4f}")
        logger.info(f"   RMSE: {format_currency(metrics['rmse'])}")
        logger.info(f"   MAE: {format_currency(metrics['mae'])}")
        logger.info(f"   MAPE: {metrics['mape']:.2f}%")
    except Exception as e:
        logger.error(f"❌ Error al evaluar modelo: {e}")
        return
    
    # 6. Cross-validation
    logger.info("🔄 Realizando validación cruzada...")
    try:
        cv_scores = trainer.cross_validate(X, y, cv_folds=5)
        if cv_scores:
            logger.info(f"✅ CV R²: {cv_scores['r2']['mean']:.4f} (+/- {cv_scores['r2']['std']:.4f})")
    except Exception as e:
        logger.warning(f"⚠️ Validación cruzada falló: {e}")
    
    # 7. Guardar modelo y preprocessor
    logger.info("💾 Guardando modelo...")
    
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        model_path = MODELS_DIR / "best_model.pkl"
        preprocessor_path = MODELS_DIR / "preprocessor.pkl"
        
        trainer.save_model(model_path)
        preprocessor.save_preprocessor(preprocessor_path)
        
        logger.info(f"✅ Modelo guardado en: {model_path}")
        logger.info(f"✅ Preprocessor guardado en: {preprocessor_path}")
    except Exception as e:
        logger.error(f"❌ Error al guardar modelo: {e}")
        return
    
    # 8. Guardar datos procesados
    logger.info("💾 Guardando datos procesados...")
    try:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        preprocessor.save_processed_data(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            PROCESSED_DATA_DIR
        )
        logger.info(f"✅ Datos procesados guardados en: {PROCESSED_DATA_DIR}")
    except Exception as e:
        logger.warning(f"⚠️ Error al guardar datos procesados: {e}")
    
    # 9. Mostrar feature importance
    if trainer.feature_importance is not None:
        logger.info("📊 Top 10 Features más importantes:")
        top_features = trainer.feature_importance.head(10)
        for _, row in top_features.iterrows():
            logger.info(f"   {row['feature']}: {row['importance_pct']:.2f}%")
    
    logger.info("=" * 80)
    logger.info("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    logger.info("=" * 80)
    
    return trainer, preprocessor, metrics


if __name__ == "__main__":
    main()