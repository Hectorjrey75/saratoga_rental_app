# test_config.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import config, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR

print("✅ Configuración cargada correctamente")
print(f"   PROJECT_ROOT: {PROJECT_ROOT}")
print(f"   RAW_DATA_DIR: {RAW_DATA_DIR}")
print(f"   PROCESSED_DATA_DIR: {PROCESSED_DATA_DIR}")
print(f"   MODELS_DIR: {MODELS_DIR}")

# Verificar que los directorios existen
print(f"\n📁 Verificando directorios:")
print(f"   RAW_DATA_DIR existe: {RAW_DATA_DIR.exists()}")
print(f"   PROCESSED_DATA_DIR existe: {PROCESSED_DATA_DIR.exists()}")
print(f"   MODELS_DIR existe: {MODELS_DIR.exists()}")