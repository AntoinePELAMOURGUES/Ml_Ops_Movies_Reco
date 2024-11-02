from typing import Union
from fastapi import FastAPI, status, Depends, HTTPException
from typing import Annotated
from sqlalchemy.orm import Session
from database import engine, SessionLocal
from models import *  # Importation de tous les modèles
from models import Base  # Importation de la base pour créer les tables

from auth import get_current_user, router as auth_router # Importation des fonctions d'authentification et du routeur
from predict import router as predict_router # Importation des fonctions de predictions
from prometheus_fastapi_instrumentator import Instrumentator


# Création de l'application FastAPI
app = FastAPI()

# Créez un instrumentateur et configurez-le
instrumentator = Instrumentator()

# Instrumenter l'application et exposer les métriques
instrumentator.instrument(app).expose(app)

# Inclusion du routeur d'authentification
app.include_router(auth_router)
app.include_router(predict_router)

# Création des tables dans la base de données
Base.metadata.create_all(bind=engine)

# Fonction pour obtenir une session de base de données
def get_db():
    db = SessionLocal()  # Crée une nouvelle session de base de données
    try:
        yield db  # Renvoie la session pour utilisation
    finally:
        db.close()  # Ferme la session à la fin

# Route pour obtenir les informations de l'utilisateur actuel
@app.get("/", status_code=status.HTTP_200_OK)
async def user(user: Annotated[dict, Depends(get_current_user)], db: Annotated[Session, Depends(get_db)]):
    if user is None:
        raise HTTPException(status_code=401, detail='Authentication Failed')  # Erreur si l'utilisateur n'est pas authentifié
    return {"User": user}  # Retourne les informations de l'utilisateur


