# Usamos la imagen oficial de Python 3.12
FROM python:3.12-slim

# Establecer directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar requirements.txt y luego instalar dependencias (esto aprovecha cache de Docker)
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copiar todo el proyecto dentro del contenedor
COPY . .

# Exponer el puerto 8000 (donde Django correrá el servidor)
EXPOSE 8080

# Comando para correr el servidor de desarrollo de Django
CMD ["python", "manage.py", "runserver", "0.0.0.0:8080"]
