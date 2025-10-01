from pydantic import BaseModel, Field, conlist, field_validator
from typing import List, Optional, Set
import json
import logging
from datetime import datetime
import pandas as pd
from rapidfuzz.distance import Levenshtein

def _find_closest_match(invalid_id: str, valid_ids_set: Set[str], max_distance: int = 2) -> Optional[str]:
    """
    Finds the closest match for an invalid ID from a set of valid IDs.

    Uses Levenshtein distance to find a unique best match within a max_distance.
    Returns the corrected ID if a unique match is found, otherwise None.
    """
    if not invalid_id:
        return None

    candidates = []
    min_dist = max_distance + 1

    # Optimization: filter by length first. Message IDs are strings of digits
    # and should have a consistent length. A small variance is allowed for typos.
    len_invalid = len(invalid_id)

    for valid_id in valid_ids_set:
        valid_id = str(valid_id)
        if abs(len_invalid - len(valid_id)) > max_distance:
            continue

        dist = Levenshtein.distance(invalid_id, valid_id)

        if dist < min_dist:
            min_dist = dist
            candidates = [valid_id]
        elif dist == min_dist:
            candidates.append(valid_id)

    # Only return a match if it's unambiguous and within the threshold
    if len(candidates) == 1 and min_dist <= max_distance:
        return candidates[0]
    
    if len(candidates) > 1:
        logging.debug(f"Ambiguous match for '{invalid_id}'. Candidates: {candidates} with distance {min_dist}. Not correcting.")

    return None

class IDValidationMixin:
    def validate_and_approx(self, valid_ids_set: Set[str], log_prefix: str = "") -> bool:
        """
        Validates and cleans the instance's message IDs in-place.
        Tries to correct invalid IDs using Levenshtein distance.
        Returns False if the Topic_ID is invalid and cannot be corrected.
        """
        topic_id = getattr(self, 'topic_id', None)
        if not topic_id or topic_id not in valid_ids_set:
            corrected_topic_id = _find_closest_match(str(topic_id), valid_ids_set)
            if corrected_topic_id:
                logging.info(f"{log_prefix} Corrected invalid Topic_ID '{topic_id}' to '{corrected_topic_id}'.")
                self.topic_id = corrected_topic_id
            else:
                header = getattr(self, 'header', topic_id)
                logging.warning(f"{log_prefix} Invalid and uncorrectable Topic_ID '{topic_id}'. Skipping thread: {header}.")
                return False
        
        # Re-fetch topic_id in case it was corrected
        topic_id = str(self.topic_id)

        # Validate answer_id if the attribute exists
        if hasattr(self, 'answer_id') and self.answer_id and self.answer_id not in valid_ids_set:
            corrected_answer_id = _find_closest_match(self.answer_id, valid_ids_set)
            if corrected_answer_id:
                logging.info(f"{log_prefix} Corrected invalid answer_id '{self.answer_id}' to '{corrected_answer_id}' for topic_id '{topic_id}'.")
                self.answer_id = corrected_answer_id
            else:
                logging.warning(f"{log_prefix} Invalid and uncorrectable answer_id '{self.answer_id}' for topic_id '{topic_id}'. Setting to None.")
                self.answer_id = None

        # Validate whole_thread messages
        if hasattr(self, 'whole_thread'):
            corrected_thread = []
            corrections = {} # 'bad_id': 'good_id'
            removals = [] # 'bad_id'

            for msg in self.whole_thread:
                if msg.message_id in valid_ids_set:
                    corrected_thread.append(msg)
                else:
                    corrected_msg_id = _find_closest_match(msg.message_id, valid_ids_set)
                    if corrected_msg_id:
                        msg.message_id = corrected_msg_id
                        corrected_thread.append(msg)
                        corrections[msg.message_id] = corrected_msg_id
                    else:
                        removals.append(msg.message_id)
            
            self.whole_thread = corrected_thread

            if corrections:
                logging.info(f"{log_prefix} Corrected {len(corrections)} Message ID(s) in whole_thread for topic_id '{topic_id}': {corrections}")
            if removals:
                logging.warning(f"{log_prefix} Removed {len(removals)} uncorrectable Message ID(s) from whole_thread for topic_id '{topic_id}': {removals}")
        
        return True
class MessageNode(BaseModel, IDValidationMixin):
    message_id: str = Field( description="Unique identifier for the message")
    parent_id: Optional[str] = Field( description="Identifier of the parent message, if any")
    image_summary: Optional[str] = Field( description="Description of image, if any")  
    # content: str = Field( description="Content of the message")

class RawThreadList(BaseModel):

    class RawThread(BaseModel, IDValidationMixin):
        topic_id: str = Field( description="ID of the first message in the thread")
        whole_thread: List[MessageNode] = Field( description="List of message nodes of this thread")

    threads: List[RawThread] = Field( description="List of threads")

    def validate_and_clean_threads(self, valid_ids_set: set, log_prefix: str = ""):
        """Filters the list of threads, keeping only those with valid IDs."""
        self.threads = [thread for thread in self.threads if thread.validate_and_approx(valid_ids_set, log_prefix)]
        return self

from enum import Enum
class ThreadStatus(str, Enum):
    NEW = 'new'
    MODIFIED = 'modified'
    PERSISTED = 'persisted'

class ModifiedRawThreadList(BaseModel):
    class ModifiedRawThread(BaseModel, IDValidationMixin):
        topic_id: str = Field( description="ID of the first message in the thread")
        whole_thread: List[MessageNode] = Field( description="List of message nodes of this thread")
        status: ThreadStatus = Field(
            default=ThreadStatus.PERSISTED, 
            description= "Status of the gathered thread: 'new' if it is completely new thread and 'modified' if the thread has both previous messages from json data and new messages from CSV table, in other cases status is 'persisted'"
            )
        # @field_validator('status')
        # @classmethod
        # def validate_label(cls, value):
        #     allowed_labels = ['new', 'modified', 'persisted']
        #     if value not in allowed_labels:
        #         raise ValueError(f"label must be one of {allowed_labels}")
        #     return value
    threads: List[ModifiedRawThread] = Field(description="List of threads")

    def validate_and_clean_threads(self, valid_ids_set: set, log_prefix: str = ""):
        """Filters the list of threads, keeping only those with valid IDs."""
        self.threads = [thread for thread in self.threads if thread.validate_and_approx(valid_ids_set, log_prefix)]
        return self

class SolutionStatus(str, Enum):
    RESOLVED = 'resolved'
    UNRESOLVED = 'unresolved'
    SUGGESTION = 'suggestion'
    OUTSIDE = 'outside'

class ThreadList(BaseModel):
    class Thread(BaseModel, IDValidationMixin):
        header: str = Field( description="General description of the problem derived by the entire conversation")
        topic_id: str = Field( description="ID of the first message in the thread")
        actual_date: datetime = Field( description="maximal datetime of messages in the thread")
        answer_id: Optional[str] = Field( description="ID of solution message")
        # whole_thread: List[str] = Field( description="List of Message IDs of this thread")
        label: SolutionStatus # str = Field(description="label indicating the resolution status: 'resolved', 'unresolved','suggestion' or 'outside'")
        solution: str = Field( description="General description of the solution to the problem derived by the entire conversation. If doesn't exist, 'N/A'")

        # @field_validator('label')
        # @classmethod
        # def validate_label(cls, value):
        #     allowed_labels = ['resolved', 'unresolved', 'suggestion', 'outside']
        #     if value not in allowed_labels:
        #         raise ValueError(f"label must be one of {allowed_labels}")
        #     return value

    threads: List[Thread]

    def validate_and_clean_threads(self, valid_ids_set: set, log_prefix: str = ""):
        """Filters the list of threads, keeping only those with valid IDs."""
        self.threads = [thread for thread in self.threads if thread.validate_and_approx(valid_ids_set, log_prefix)]
        return self

class TechnicalTopics(BaseModel):
    """List of technical topic IDs."""
    technical_topics: List[str] = Field(description="List of Topic IDs for technical threads.")


class RevisedStatus(str, Enum):
    IMPROVED = 'improved'
    CHANGED = 'changed'
    MINORCHANGES = 'minor'

class RevisedList(BaseModel):
    """
    results of comparison of old and modified solutions
    """
    class RevisedSolution(BaseModel):
        """
        single comparison of old and modified solution
        """
        topic_id: str = Field(..., description="solution ID")
        label: RevisedStatus = Field(RevisedStatus.MINORCHANGES, description="classification result label")
        # @field_validator('label')
        # @classmethod
        # def validate_label(cls, value):
        #     allowed_labels = ['improved', 'changed', 'persisted']
        #     if value not in allowed_labels:
        #         raise ValueError(f"label must be one of {allowed_labels}")
        #     return value

    comparisions: List[RevisedSolution] = Field(..., description="list of classification results")
