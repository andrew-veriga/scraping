import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import json

from app.main import app
from app.main import process_batch

client = TestClient(app)

@pytest.fixture
def mock_services(monkeypatch):
    """Mocks all services and utilities called by the main endpoints."""
    # Mock data_loader
    mock_load_data = MagicMock(return_value=pd.DataFrame({
        'DateTime': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-10T12:00:00']).tz_localize('UTC'),
        'Message ID': ['1', '2', '3'],
        'Author ID': ['101', '102', '101'],
        'Content': ['a', 'b', 'c'],
        'Referenced Message ID': ['', '1', ''],
        'DatedMessage': ['msg1', 'msg2', 'msg3']
    }))
    monkeypatch.setattr("app.services.data_loader.load_and_preprocess_data", mock_load_data)

    # Mock thread_service
    monkeypatch.setattr("app.services.thread_service.first_thread_gathering", MagicMock(return_value="step1.json"))
    monkeypatch.setattr("app.services.thread_service.filter_technical_topics", MagicMock(return_value="tech.json"))
    monkeypatch.setattr("app.services.thread_service.generalization_solution", MagicMock(return_value="solutions.json"))
    # This function is missing from the provided code, so it must be mocked.
    monkeypatch.setattr("app.services.solution_service.new_solutions_revision_and_add", MagicMock())

    # Mock file_utils
    # Note: get_end_date_from_solutions needs to return a timezone-aware datetime
    mock_load_solutions = MagicMock(return_value={'1': {'Actual_Date': '2023-01-05T00:00:00+00:00'}})
    monkeypatch.setattr("app.utils.file_utils.load_solutions_dict", mock_load_solutions)
    monkeypatch.setattr("app.utils.file_utils.save_solutions_dict", MagicMock())
    monkeypatch.setattr("app.utils.file_utils.create_dict_from_list", MagicMock(return_value={'2': {'Actual_Date': '2023-01-06T00:00:00+00:00'}}))
    monkeypatch.setattr("app.utils.file_utils.get_end_date_from_solutions", MagicMock(return_value=pd.Timestamp('2023-01-05 00:00:00+0000', tz='UTC')))

    # Mock process_batch itself for the next_batch endpoint test
    mock_process_batch = MagicMock()
    monkeypatch.setattr("app.main.process_batch", mock_process_batch)

    # Mock json.load for process_first_batch
    mock_json_load = MagicMock(return_value=[{'topic_id': '1'}])
    m_open = patch('builtins.open', MagicMock())
    m_json_load = patch('json.load', mock_json_load)

    with m_open, m_json_load:
        yield {
            "load_data": mock_load_data,
            "process_batch": mock_process_batch,
            "load_solutions": mock_load_solutions,
        }

def test_process_first_batch_success(mock_services):
    response = client.post("/process-first-batch")
    assert response.status_code == 200
    assert response.json() == {"message": "First batch processed successfully"}
    assert mock_services["load_data"].called

def test_process_first_batch_exception(mock_services):
    mock_services["load_data"].side_effect = Exception("Test Error")
    response = client.post("/process-first-batch")
    assert response.status_code == 500
    assert response.json() == {"detail": "Test Error"}

def test_process_next_batches_success(mock_services):
    response = client.post("/process-next-batch")
    assert response.status_code == 200
    assert response.json() == {"message": "Next batch processed successfully"}
    mock_services["load_data"].assert_called_once()
    mock_services["load_solutions"].assert_called_once()
    # The loop should run and call process_batch when it finds data
    assert mock_services["process_batch"].call_count > 0

def test_process_next_batches_exception(mock_services):
    mock_services["load_data"].side_effect = Exception("Test Error")
    response = client.post("/process-next-batch")
    assert response.status_code == 500
    assert response.json() == {"detail": "Test Error"}

def test_process_next_batches_no_existing_solutions(mock_services, monkeypatch):
    """Tests processing the next batch when no solutions file exists yet."""
    monkeypatch.setattr("app.utils.file_utils.get_end_date_from_solutions", MagicMock(return_value=None))
    # With no existing solutions, it should start from the beginning of messages
    # and process batches until it's caught up.
    response = client.post("/process-next-batch")
    assert response.status_code == 200
    assert response.json() == {"message": "Next batch processed successfully"}
    mock_services["load_data"].assert_called_once()
    mock_services["load_solutions"].assert_called_once()
    # It should have found messages and called process_batch
    assert mock_services["process_batch"].call_count > 0

def test_process_next_batches_no_new_messages(mock_services, monkeypatch):
    """Tests processing when the solutions are already up-to-date."""
    # Set the latest solution date to be after the last message
    last_message_date = mock_services["load_data"].return_value['DateTime'].max()
    monkeypatch.setattr("app.utils.file_utils.get_end_date_from_solutions", MagicMock(return_value=last_message_date))

    response = client.post("/process-next-batch")
    assert response.status_code == 200
    assert response.json() == {"message": "Next batch processed successfully"}
    mock_services["load_data"].assert_called_once()
    mock_services["load_solutions"].assert_called_once()
    # No new messages should be found, so process_batch should not be called
    assert mock_services["process_batch"].call_count == 0

def test_get_solutions_success(mock_services):
    """Tests successfully retrieving the solutions."""
    response = client.get("/solutions")
    assert response.status_code == 200
    assert response.json() == {'1': {'Actual_Date': '2023-01-05T00:00:00+00:00'}}
    mock_services["load_solutions"].assert_called_once()

def test_get_solutions_not_found(mock_services):
    """Tests the response when the solutions file doesn't exist."""
    mock_services["load_solutions"].side_effect = FileNotFoundError("File not found")
    response = client.get("/solutions")
    assert response.status_code == 404
    assert response.json() == {"detail": "Solutions file not found. Please run a batch process first."}