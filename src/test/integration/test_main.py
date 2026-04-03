# src/test/integration/test_main.py
import pytest
import requests

BASE_URL = "http://localhost:8000"

def test_root_redirect():
    response = requests.get(BASE_URL + "/", allow_redirects=False)
    assert response.status_code in [302, 307]

def test_get_predictions_not_found():
    response = requests.get(BASE_URL + "/predictions?model_name=inexistant")
    assert response.status_code == 404

def test_get_predictions():
    response = requests.get(BASE_URL + "/predictions")
    assert response.status_code in [200, 404]  

def test_get_version_no_champion():
    response = requests.get(BASE_URL + "/version")
    assert response.status_code in [200, 404]

def test_get_combined_missing_params():
    response = requests.get(BASE_URL + "/predictions/combined")
    assert response.status_code == 422  