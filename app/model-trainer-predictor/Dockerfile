FROM python:3.12

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier des dépendances
COPY ./requirements.txt /app/requirements.txt

# Installer les dépendances
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copier tout le code de l'application
COPY ./app /app

# Définir un point d'entrée par défaut qui ne fait rien
CMD ["tail", "-f", "/dev/null"]

