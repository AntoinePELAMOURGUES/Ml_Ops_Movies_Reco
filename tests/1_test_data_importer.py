import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
import sys

# Ajouter le chemin src au sys.path pour permettre l'importation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from import_raw_data import import_raw_data

@pytest.fixture
def mock_requests_get():
    with patch('requests.get') as mock_get:
        yield mock_get

@pytest.fixture
def mock_check_existing():
    with patch('data_importer.check_existing_folder') as mock_check_folder, \
         patch('data_importer.check_existing_file') as mock_check_file:
        yield mock_check_folder, mock_check_file

def test_import_raw_data(mock_requests_get, mock_check_existing):
    mock_check_folder, mock_check_file = mock_check_existing

    # Configuration des mocks
    mock_check_folder.return_value = True  # Simule que le dossier existe déjà
    mock_check_file.return_value = False  # Simule que le fichier n'existe pas

    # Simuler une réponse réussie de requests.get()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "dummy content"
    mock_requests_get.return_value = mock_response

    raw_data_relative_path = "./data/raw"
    filenames = ["test_file.csv"]
    bucket_folder_url = "https://example.com/"

    import_raw_data(raw_data_relative_path, filenames, bucket_folder_url)

    # Vérifier que requests.get() a été appelé avec la bonne URL
    mock_requests_get.assert_called_once_with("https://example.com/test_file.csv")

    # Vérifier que le fichier a été ouvert en mode écriture binaire
    mock_open.assert_called_once_with(os.path.join(raw_data_relative_path, "test_file.csv"), "wb")

    # Vérifier que le contenu a été écrit dans le fichier
    handle = mock_open()
    handle().write.assert_called_once_with("dummy content".encode('utf-8'))

def test_import_raw_data_error(mock_requests_get, mock_check_existing):
    mock_check_folder, mock_check_file = mock_check_existing

    # Configuration des mocks pour simuler une erreur lors du téléchargement
    mock_check_folder.return_value = True
    mock_check_file.return_value = False

    # Simuler une réponse d'erreur de requests.get()
    mock_response = MagicMock()
    mock_response.status_code = 404  # Simule une erreur 404
    mock_requests_get.return_value = mock_response

    raw_data_relative_path = "./data/raw"
    filenames = ["test_file.csv"]
    bucket_folder_url = "https://example.com/"

    with pytest.raises(Exception) as excinfo:  # Remplacez Exception par le type d'exception approprié.
        import_raw_data(raw_data_relative_path, filenames, bucket_folder_url)

        # Vérifier que le bon message d'erreur a été enregistré dans les logs (si applicable)
        assert "Error accessing the object https://example.com/test_file.csv:" in str(excinfo.value)

if __name__ == '__main__':
    pytest.main()