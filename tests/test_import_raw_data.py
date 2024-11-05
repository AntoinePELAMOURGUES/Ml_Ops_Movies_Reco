import os
import sys
import pytest
from unittest import mock

# Ajouter le chemin du dossier src au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from import_raw_data import import_raw_data

# Assurez-vous que le dossier de test existe
TEST_RAW_DATA_PATH = "./tests/data/raw"
BUCKET_FOLDER_URL = "https://mlops-project-db.s3.eu-west-1.amazonaws.com/movie_recommandation/"
FILENAMES = ["genome-scores.csv", "genome-tags.csv"]

# Mock des fonctions check_existing_file et check_existing_folder
@mock.patch('check_structure.check_existing_file')
@mock.patch('check_structure.check_existing_folder')
@mock.patch('requests.get')
def test_import_raw_data(mock_get, mock_check_existing_folder, mock_check_existing_file):
    # Configuration des mocks
    mock_check_existing_folder.return_value = False  # Simule que le dossier n'existe pas
    mock_check_existing_file.return_value = False  # Simule que le fichier n'existe pas

    # Simuler une réponse réussie de requests.get
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.text = "mocked content"
    mock_get.return_value = mock_response

    # Appel de la fonction à tester
    import_raw_data(TEST_RAW_DATA_PATH, FILENAMES, BUCKET_FOLDER_URL)

    # Vérifiez que le dossier a été créé
    assert os.path.exists(TEST_RAW_DATA_PATH)

    # Vérifiez que requests.get a été appelé avec les bonnes URL
    for filename in FILENAMES:
        expected_url = os.path.join(BUCKET_FOLDER_URL, filename)
        mock_get.assert_any_call(expected_url)

    # Vérifiez que le contenu a été écrit dans les fichiers (vous pouvez également vérifier leur existence)
    for filename in FILENAMES:
        output_file_path = os.path.join(TEST_RAW_DATA_PATH, filename)
        assert os.path.isfile(output_file_path)

# Nettoyage après le test
@pytest.fixture(scope='module', autouse=True)
def cleanup():
    yield
    if os.path.exists(TEST_RAW_DATA_PATH):
        for filename in FILENAMES:
            file_path = os.path.join(TEST_RAW_DATA_PATH, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(TEST_RAW_DATA_PATH)