#!/bin/bash
set -e

echo "🚀 Iniciando Streamlit..."
exec streamlit run app/main.py \
    --server.headless true \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false \
    --logger.level info