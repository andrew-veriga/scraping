import json
import os
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import pytest
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from app.services import thread_processor
from app.services import gemini_service

# --- Mock Pydantic Models ---
# These would normally be in app.models.pydantic_models. We mock them here
# because the file is not provided.
class MockMessage(BaseModel):
    Message_ID: str
    Author_ID: str
    Content: str

class MockThread(BaseModel):
    Topic_ID: str
    Whole_thread: List[str]
    Whole_thread_formatted: Optional[List[MockMessage]] = None
    status: Optional[str] = None

class MockThreadList(BaseModel):
    threads: List[MockThread]

class MockTechnicalTopics(BaseModel):
    technical_topics: List[str]

class MockSolution(BaseModel):
    Topic_ID: str
    Problem_Statement: str
    Solution: str
    Status: str
    Actual_Date: datetime

class MockSolutionList(BaseModel):
    threads: List[MockSolution]

# --- Fixtures for Mock Gemini Responses ---

@pytest.fixture
def mock_gemini_response_step1():
    thread = MockThread(Topic_ID="1", Whole_thread=["1", "2", "3"])
    thread_list = MockThreadList(threads=[thread])
    return gemini_service._MockParsedResponse(thread_list)

@pytest.fixture
def mock_gemini_response_step2():
    tech_topics = MockTechnicalTopics(technical_topics=["1"])
    return gemini_service._MockParsedResponse(tech_topics)

@pytest.fixture
def mock_gemini_response_step3():
    solution = MockSolution(
        Topic_ID="1",
        Problem_Statement="Problem",
        Solution="Solution",
        Status="resolved",
        Actual_Date=datetime.now()
    )
    solution_list = MockSolutionList(threads=[solution])
    return gemini_service._MockParsedResponse(solution_list)

# --- Tests ---

def test_first_thread_gathering(sample_messages_df, mock_gemini_response_step1, tmp_path):
    save_path = str(tmp_path)

    with patch('app.services.gemini_service.generate_content', return_value=mock_gemini_response_step1) as mock_generate, \
         patch('builtins.open', mock_open()) as mock_file, \
         patch('json.dump') as mock_json_dump:

        result_path = thread_processor.first_thread_gathering(sample_messages_df, save_path)

        mock_generate.assert_called_once()
        assert "first_group" in os.path.basename(result_path)
        mock_file.assert_called_once_with(result_path, 'w')
        mock_json_dump.assert_called_once()
        dumped_data = mock_json_dump.call_args[0][0]
        assert len(dumped_data) == 1
        assert dumped_data[0]['Topic_ID'] == '1'

def test_illustrated_threads(sample_messages_df):
    threads_json_data = [{'Topic_ID': '1', 'Whole_thread': ['1', '2']}]

    enriched_data = thread_processor.illustrated_threads(threads_json_data, sample_messages_df)

    assert len(enriched_data) == 1
    thread = enriched_data[0]
    assert 'Whole_thread_formatted' in thread
    assert len(thread['Whole_thread_formatted']) == 2
    assert thread['Whole_thread_formatted'][0]['Message_ID'] == '1'
    assert thread['Whole_thread_formatted'][0]['Author_ID'] == '101'
    assert "How do I stake SUI?" in thread['Whole_thread_formatted'][0]['Content']

def test_filter_technical_topics(sample_messages_df, mock_gemini_response_step2, tmp_path):
    save_path = str(tmp_path)
    threads_data = [{'Topic_ID': '1', 'Whole_thread': ['1', '2']}, {'Topic_ID': '4', 'Whole_thread': ['4', '5']}]
    input_filename = os.path.join(save_path, "input.json")
    with open(input_filename, 'w') as f:
        json.dump(threads_data, f)

    # We mock open with the real implementation for the initial read, but then patch the write
    with patch('app.services.gemini_service.generate_content', return_value=mock_gemini_response_step2) as mock_generate, \
         patch('builtins.open', mock_open(read_data=json.dumps(threads_data))) as mock_file:

        result_path = thread_processor.filter_technical_topics(input_filename, "first", sample_messages_df, save_path)

        mock_generate.assert_called_once()
        assert "first_technical" in os.path.basename(result_path)
        
        handle = mock_file()
        handle.write.assert_called_once()
        written_content = handle.write.call_args[0][0]
        written_json = json.loads(written_content)
        assert len(written_json) == 1
        assert written_json[0]['Topic_ID'] == '1'

def test_generalization_solution(mock_gemini_response_step3, tmp_path):
    save_path = str(tmp_path)
    technical_threads = [{'Topic_ID': '1', 'Whole_thread_formatted': [{'Message_ID': '1', 'Author_ID': '101', 'Content': '...'}]}]
    input_filename = os.path.join(save_path, "tech_input.json")
    with open(input_filename, 'w') as f:
        json.dump(technical_threads, f)

    with patch('app.services.gemini_service.generate_content', return_value=mock_gemini_response_step3) as mock_generate, \
         patch('builtins.open', mock_open(read_data=json.dumps(technical_threads))) as mock_file, \
         patch('json.dump') as mock_json_dump:

        result_path = thread_processor.generalization_solution(input_filename, "first", save_path)

        mock_generate.assert_called_once()
        assert "first_solutions" in os.path.basename(result_path)
        mock_file.assert_called_with(result_path, 'w')
        mock_json_dump.assert_called_once()
        dumped_data = mock_json_dump.call_args[0][0]
        assert len(dumped_data) == 1
        assert dumped_data[0]['Topic_ID'] == '1'