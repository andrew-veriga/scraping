from pydantic import BaseModel, Field, conlist, field_validator
from typing import List, Optional
import json
from datetime import datetime
import pandas as pd

class RawThreadList(BaseModel):

    class RawThread(BaseModel):
        Topic_ID: str = Field( description="Id of the first message in the thread")
        Whole_thread: List[str] = Field( description="List of Message IDs of this thread")

    threads: List[RawThread] = Field( description="List of threads")

class ModifiedRawThreadList(BaseModel):
    class ModifiedRawThread(BaseModel):
        Topic_ID: str = Field( description="Id of the first message in the thread")
        Whole_thread: List[str] = Field( description="List of Message IDs of this thread")
        status: str = Field(default='persisted',description= "Status of the gathered thread: 'new' if it is completely new thread and 'modified' if the thread has both previous messages from json data and new messages from CSV table, in other cases status is 'persisted'")
        @field_validator('status')
        @classmethod
        def validate_label(cls, value):
            allowed_labels = ['new', 'modified', 'persisted']
            if value not in allowed_labels:
                raise ValueError(f"Label must be one of {allowed_labels}")
            return value
    threads: List[ModifiedRawThread] = Field( description="List of threads")

class ThreadList(BaseModel):
    class Thread(BaseModel):
        Header: str = Field( description="General description of the problem derived by the entire conversation")
        Topic_ID: str = Field( description="Id of the first message in the thread")
        Actual_Date: datetime = Field( description="DateTime of the last message in the thread")
        Answer_ID: Optional[str] = Field( description="Id of solution message")
        Whole_thread: List[str] = Field( description="List of 'Message ID' of this thread")
        Label: str = Field(description="Label indicating the resolution status: 'resolved', 'unresolved','suggestion' or 'outside'")
        Solution: str = Field( description="General description of the solution to the problem derived by the entire conversation. If doesn't exist, 'N/A'")

        @field_validator('Label')
        @classmethod
        def validate_label(cls, value):
            allowed_labels = ['resolved', 'unresolved', 'suggestion', 'outside']
            if value not in allowed_labels:
                raise ValueError(f"Label must be one of {allowed_labels}")
            return value

    threads: List[Thread]

class TechnicalTopics(BaseModel):
    """List of technical topic IDs."""
    technical_topics: List[str] = Field(description="List of Topic IDs for technical threads.")

class RevisedList(BaseModel):
    """
    results of comparison of old and modified solutions
    """
    class RevisedSolution(BaseModel):
        """
        single comparison of old and modified solution
        """
        Topic_ID: str = Field(..., description="solution ID")
        Label: str = Field(..., description="classification result label")
        @field_validator('Label')
        @classmethod
        def validate_label(cls, value):
            allowed_labels = ['improved', 'changed', 'persisted']
            if value not in allowed_labels:
                raise ValueError(f"Label must be one of {allowed_labels}")
            return value

    comparisions: List[RevisedSolution] = Field(..., description="list of classification results")
