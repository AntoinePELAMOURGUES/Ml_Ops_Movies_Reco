text

# API Documentation

## Introduction

Cette API permet la gestion des utilisateurs avec des fonctionnalités d'inscription et d'authentification. Elle utilise FastAPI et SQLAlchemy pour gérer les requêtes et interagir avec une base de données SQLite.

## Endpoints

### 1. Create User

**POST** `/auth/`

- **Description** : Crée un nouvel utilisateur dans la base de données.
- **Request Body** :
  ```json
  {
    "username": "nouvel_utilisateur",
    "password": "mot_de_passe"
  }
  ```

Responses :

- 201 Created : Utilisateur créé avec succès.
- 400 Bad Request : "Username already registered" si le nom d'utilisateur existe déjà.

### 2. Login for Access Token

**POST** '/auth/token'

- **Description** : Permet à un utilisateur de se connecter et d'obtenir un token d'accès.
- **Request Body** :

```json
{
  "username": "utilisateur",
  "password": "mot_de_passe"
}
```

- **Responses** :
- 200 OK : Retourne un token d'accès.

```json
{
  "access_token": "votre_token",
  "token_type": "bearer"
}
```

- 401 Unauthorized : "Could not validate user." si les informations d'identification sont incorrectes.

### 3. Get User Information

**GET** '/'

- **Description** : Permet à un utilisateur authentifié d'accéder à ses informations.
- **Headers** :
- text
- Authorization: Bearer votre_token

**Responses** :
200 OK : Retourne les informations de l'utilisateur.

```json
{
  "User": {
    "username": "utilisateur",
    "id": 1
  }
}
```

- 401 Unauthorized : "Authentication Failed." si le token est invalide ou a expiré.

## Flux d'Utilisation

- Inscription : L'utilisateur s'inscrit en envoyant ses informations à /auth/. Si l'inscription réussit, il peut se connecter.
- Connexion : L'utilisateur se connecte en envoyant ses informations à /auth/token. Si la connexion réussit, il reçoit un token d'accès.
- Accès aux Ressources : L'utilisateur utilise le token d'accès pour accéder à des ressources protégées, comme /, en l'incluant dans l'en-tête Authorization.

## Conclusion

Ce processus permet de gérer l'inscription et l'authentification des utilisateurs de manière sécurisée en utilisant des tokens JWT, ce qui est courant dans les applications modernes.
