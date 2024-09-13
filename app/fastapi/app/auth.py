from datetime import timedelta, datetime
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette import status
from .database import SessionLocal
from .models import User
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import jwt, JWTError
import os
from dotenv import load_dotenv

# Création d'un routeur pour gérer les routes d'authentification
router = APIRouter(
    prefix='/auth',  # Préfixe pour toutes les routes dans ce routeur
    tags=['auth']    # Tag pour la documentation
)
# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

# Clé secrète pour le JWT (à remplacer par une variable d'environnement en production)
SECRET_KEY = os.getenv('SECRET_KEY')
ALGORITHM = os.getenv('ALGORITHM')  # Algorithme utilisé pour encoder le JWT

# Contexte de hachage pour le mot de passe
bcrypt_context = CryptContext(schemes=['bcrypt'], deprecated='auto')

# Définition du schéma de sécurité pour le token
oauth2_bearer = OAuth2PasswordBearer(tokenUrl='auth/token')

# Modèle Pydantic pour la création d'un utilisateur
class CreateUserRequest(BaseModel):
    username: str  # Nom d'utilisateur
    password: str  # Mot de passe

# Modèle Pydantic pour le token d'accès
class Token(BaseModel):
    access_token: str  # Le token d'accès
    token_type: str    # Type de token (généralement "bearer")

# Fonction pour obtenir une session de base de données
def get_db():
    db = SessionLocal()  # Crée une nouvelle session de base de données
    try:
        yield db  # Renvoie la session pour utilisation
    finally:
        db.close()  # Ferme la session à la fin


# Route pour créer un nouvel utilisateur
@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_user(db: Annotated[Session, Depends(get_db)], create_user_request: CreateUserRequest):
    # Vérifiez si l'utilisateur existe déjà
    existing_user = db.query(User).filter(User.username == create_user_request.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")  # Erreur si l'utilisateur existe

    # Créer le modèle utilisateur
    create_user_model = User(
        username=create_user_request.username,
        hashed_password=bcrypt_context.hash(create_user_request.password),  # Hachage du mot de passe
    )

    db.add(create_user_model)  # Ajoute l'utilisateur à la session
    db.commit()  # Commit les changements dans la base de données
    db.refresh(create_user_model)  # Rafraîchit l'instance pour obtenir l'ID
    return create_user_model  # Retourne le modèle utilisateur créé

# Route pour obtenir un token d'accès
@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()], db: Annotated[Session, Depends(get_db)]):
    user = authenticate_user(form_data.username, form_data.password, db)  # Authentifie l'utilisateur
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Could not validate user.')  # Erreur si l'authentification échoue

    token = create_access_token(user.username, user.id, timedelta(minutes=30))  # Crée un token d'accès
    return {"access_token": token, "token_type": "bearer"}  # Retourne le token et son type

# Fonction pour authentifier un utilisateur
def authenticate_user(username: str, password: str, db: Session):
    user = db.query(User).filter(User.username == username).first()  # Récupère l'utilisateur par nom d'utilisateur
    if not user:
        return False  # Retourne False si l'utilisateur n'existe pas
    if not bcrypt_context.verify(password, user.hashed_password):  # Vérifie le mot de passe
        return False  # Retourne False si le mot de passe est incorrect
    return user  # Retourne l'utilisateur si l'authentification réussit

# Fonction pour créer un token d'accès
def create_access_token(username: str, user_id: int, expires_delta: timedelta):
    encode = {'sub': username, 'id': user_id}  # Charge utile du token
    expires = datetime.utcnow() + expires_delta  # Définit la date d'expiration
    encode.update({'exp': expires})  # Ajoute la date d'expiration à la charge utile
    return jwt.encode(encode, SECRET_KEY, algorithm=ALGORITHM)  # Encode le token

# Fonction pour obtenir l'utilisateur actuel à partir du token
async def get_current_user(token: Annotated[str, Depends(oauth2_bearer)]):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])  # Décode le token
        username: str = payload.get('sub')  # Récupère le nom d'utilisateur
        user_id: int = payload.get('id')  # Récupère l'ID de l'utilisateur
        if username is None or user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Could not validate user.')  # Erreur si les données sont manquantes
        return {'username': username, 'id': user_id}  # Retourne les données de l'utilisateur
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Could not validate user')  # Erreur si le token est invalide