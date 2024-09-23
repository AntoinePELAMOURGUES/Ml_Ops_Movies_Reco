from sqlalchemy import create_engine  # Importation de la fonction pour créer un engine
from sqlalchemy.orm import sessionmaker  # Importation de la fonction pour créer des sessions
from sqlalchemy.ext.declarative import declarative_base  # Importation de la classe de base déclarative

# URL de la base de données SQLite
SQLALCHEMY_DATABASE_URL = 'sqlite:////app/user_db/movies_app_users.db'

# Création de l'engine de la base de données
# connect_args={'check_same_thread': False} est utilisé pour permettre l'accès à la base de données depuis plusieurs threads
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={'check_same_thread': False})

# Création d'une session locale pour interagir avec la base de données
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Création de la classe de base pour les modèles SQLAlchemy
Base = declarative_base()