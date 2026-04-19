# Usar imagen oficial de Python
FROM python:3.10-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (para aprovechar caché de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Crear directorios necesarios
RUN mkdir -p data/raw data/processed data/models logs


# Copiar script de entrada
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
# Exponer puerto de Streamlit
EXPOSE 8501

# Configurar Streamlit para ejecutarse en modo headless
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Instruccion pendiente
# Usar el script de entrada
ENTRYPOINT ["/entrypoint.sh"]

# Comando para ejecutar la aplicación
#CMD ["streamlit", "run", "app/main.py", "--server.headless", "true", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]