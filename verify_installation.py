# Script de verificación rápida
# verify_installation.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def verify_installation():
    """Verificar que todos los componentes están instalados."""
    
    checks = []
    
    # 1. Verificar imports
    try:
        from src.config.settings import config
        checks.append(("✅ Config", True))
    except Exception as e:
        checks.append((f"❌ Config: {e}", False))
    
    try:
        from src.data.preprocessing import DataPreprocessor
        checks.append(("✅ Preprocessing", True))
    except Exception as e:
        checks.append((f"❌ Preprocessing: {e}", False))
    
    try:
        from src.models.model_training import ModelTrainer
        checks.append(("✅ Model Training", True))
    except Exception as e:
        checks.append((f"❌ Model Training: {e}", False))
    
    try:
        from src.prediction.prediction import RentalPricePredictor
        checks.append(("✅ Prediction", True))
    except Exception as e:
        checks.append((f"❌ Prediction: {e}", False))
    
    # 2. Verificar archivos de datos
    data_path = PROJECT_ROOT / "data" / "raw" / "SaratogaHouses.csv"
    if data_path.exists():
        checks.append(("✅ Dataset", True))
    else:
        checks.append(("❌ Dataset no encontrado", False))
    
    # 3. Verificar modelo entrenado
    model_path = PROJECT_ROOT / "data" / "models" / "best_model.pkl"
    if model_path.exists():
        checks.append(("✅ Modelo entrenado", True))
    else:
        checks.append(("⚠️ Modelo no entrenado aún", True))
    
    print("=" * 60)
    print("🔍 VERIFICACIÓN DE INSTALACIÓN")
    print("=" * 60)
    
    for check, status in checks:
        print(check)
    
    all_passed = all(status for _, status in checks)
    
    if all_passed:
        print("\n✅ ¡Todo está correctamente instalado!")
        print("\n📋 Siguientes pasos:")
        if not model_path.exists():
            print("   1. Ejecuta 'python train_model.py' para entrenar el modelo")
        print("   2. Ejecuta 'streamlit run app/main.py' para iniciar la app")
    else:
        print("\n❌ Hay problemas con la instalación")
    
    print("=" * 60)


if __name__ == "__main__":
    verify_installation()