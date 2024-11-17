# Usar una imagen oficial de Python como imagen base
FROM python:3.10.11

# Establecer el directorio de trabajo en el contenedor
WORKDIR /usr/src/app

# Copiar el contenido del directorio actual en el contenedor en /usr/src/app
COPY . .

# Instalar cualquier otro paquete necesario especificado en requirements.txt
# Se usa --no-cache-dir para no almacenar los archivos de caché de pip, reduciendo el tamaño de la imagen
RUN pip install --no-cache-dir -r requirements.txt

# Hacer disponible el puerto 8501 al mundo exterior a este contenedor
# Esto no publica el puerto, solo indica que el puerto está destinado a ser publicado
EXPOSE 8501

# Ejecutar trabajo_MLE.py cuando se inicie el contenedor
CMD ["streamlit", "run", "trabajo_MLE.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
