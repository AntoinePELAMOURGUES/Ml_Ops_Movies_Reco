import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from pymongo import MongoClient
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import pickle

def read_ratings(ratings_csv: str, data_dir: str = "/opt/airflow/data/raw") -> pd.DataFrame:
    """Reads the CSV file containing movie ratings."""
    data = pd.read_csv(os.path.join(data_dir, ratings_csv))
    print("Dataset ratings loaded")
    return data

def create_X(df):
    """Generates a sparse user-item rating matrix."""
    M = df['userId'].nunique()
    N = df['movieId'].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(M))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(N))))

    user_index = [user_mapper[i] for i in df['userId']]
    item_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (user_index, item_index)), shape=(M, N))

    return X

def train_model(X, k=10):
    """Trains the KNN model on the training data."""
    X = X.T  # Transpose to have users in rows
    kNN = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric='cosine')

    model = kNN.fit(X)

    return model

def save_model(model, filepath: str) -> None:
    """Sauvegarde le modèle entraîné dans un fichier."""
    directory = os.path.join(filepath, 'model_knn.pkl')
    with open(directory, 'wb') as file:
        pickle.dump(model, file)
        print(f'Modèle sauvegardé sous {filepath}/model.pkl')

def run_training(**kwargs):
    """Main function to train the model."""

    # Load data
    ratings = read_ratings('ratings.csv')

    # Create sparse matrix from ratings
    X = create_X(ratings)

    # Train KNN model
    model_knn = train_model(X)

    save_model(model_knn, '/opt/airflow/model/')

# Define Airflow DAG
my_dag = DAG(
    dag_id='KNN_train_model',
    description='KNN Model for Movie Recommendation',
    tags=['reco_movies'],
    schedule_interval='@daily',
    default_args={
        'owner': 'airflow',
        'start_date': datetime(2024, 10, 30),
    }
)

# Create a task to train the model and evaluate its performance
train_task = PythonOperator(
   task_id='train_model',
   python_callable=run_training,
   dag=my_dag,
)

if __name__ == "__main__":
   my_dag.cli()