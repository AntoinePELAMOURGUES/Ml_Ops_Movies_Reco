from database import Base
from sqlalchemy import Column, Integer, String

# définition du modèle User avec les champs appropriés pour base de données movies_app_users
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True)
    hashed_password = Column(String)