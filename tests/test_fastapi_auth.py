import os
import sys
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
# Ajouter le chemin du dossier src au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../fastapi/app')))
from database import Base
from main import app, get_db

# Configuration de la base de données pour les tests
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Créer les tables de test
Base.metadata.create_all(bind=engine)

# Dépendance de test pour obtenir une session de test
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

# Remplacer la dépendance dans l'application FastAPI
app.dependency_overrides[get_db] = override_get_db

# Initialiser le client de test
client = TestClient(app)

@pytest.fixture(scope="module")
def setup_database():
    # Créer un utilisateur de test avant d'exécuter les tests
    db = TestingSessionLocal()
    user_data = {
        "username": "testuser",
        "password": "TestPassword123!"
    }
    hashed_password = bcrypt_context.hash(user_data["password"])
    db.execute(
        User.__table__.insert().values(username=user_data["username"], hashed_password=hashed_password)
    )
    db.commit()
    yield db  # Yield permet d'utiliser la base de données pendant les tests

    # Nettoyer après les tests
    db.execute(User.__table__.delete().where(User.username == user_data["username"]))
    db.commit()
    db.close()

def test_create_user():
    response = client.post("/auth/", json={"username": "newuser", "password": "NewPassword123!"})

    assert response.status_code == 201
    assert response.json()["username"] == "newuser"

def test_create_user_with_existing_username(setup_database):
    response = client.post("/auth/", json={"username": "testuser", "password": "TestPassword123!"})

    assert response.status_code == 400
    assert response.json()["detail"] == "Username already registered"

def test_login_for_access_token(setup_database):
    response = client.post("/auth/token", data={"username": "testuser", "password": "TestPassword123!"})

    assert response.status_code == 200
    assert "access_token" in response.json()

def test_login_invalid_password(setup_database):
    response = client.post("/auth/token", data={"username": "testuser", "password": "WrongPassword"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Mot de passe incorrect"

def test_login_nonexistent_user():
    response = client.post("/auth/token", data={"username": "nonexistent", "password": "SomePassword"})

    assert response.status_code == 404
    assert response.json()["detail"] == "Utilisateur non trouvé"