import os

from google import genai
from google.genai import types #import GenerationConfig, ThinkingConfig
from app.models.pydantic_models import RawThreadList, ModifiedRawThreadList, ThreadList, TechnicalTopics, RevisedList
import logging

#load_dotenv(dotenv_path)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # Use logging.critical for errors that prevent the app from starting.
    logging.critical("GEMINI_API_KEY environment variable is not set.")
    raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")


gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)
model_name = "gemini-2.5-flash"


config_step1 = types.GenerateContentConfig( 
    seed=42,
    temperature=1.0,
    response_mime_type="application/json",
    thinking_config=types.ThinkingConfig(thinking_budget=2000),
    response_schema=RawThreadList
    )
config_addition_step1 = types.GenerateContentConfig(
    seed=42,
    temperature=1.0,
    response_mime_type="application/json",
    thinking_config=types.ThinkingConfig(thinking_budget=2000),
    response_schema=ModifiedRawThreadList
    )

config_step2 = types.GenerateContentConfig(
    seed=42,
    temperature= 1.0,
    response_mime_type= "application/json",
    thinking_config=types.ThinkingConfig(thinking_budget=1000),
    response_schema=TechnicalTopics
    )
solution_config = types.GenerateContentConfig(
    seed=42,
    temperature= 0.5,
    response_mime_type= "application/json",
    thinking_config=types.ThinkingConfig(thinking_budget=2500),
    response_schema=ThreadList
    )

revision_config = types.GenerateContentConfig(
    seed=42,
    temperature= 1.0,
    response_mime_type= "application/json",
    thinking_config=types.ThinkingConfig(thinking_budget=2000),
    response_schema=RevisedList
)



from typing import Set, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core.exceptions import ServiceUnavailable
from google.genai.errors import ServerError

@retry(
    wait=wait_exponential(multiplier=2, min=5, max=120),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((ServiceUnavailable, ServerError)),
    before_sleep=lambda retry_state: logging.warning(
        f"Retrying API call due to: {retry_state.outcome.exception()}. "
        f"Attempt #{retry_state.attempt_number}. Waiting {retry_state.next_action.sleep:.2f} seconds."
    )
)
def generate_content(contents, config: types.GenerateContentConfig, valid_ids_set: Optional[Set[str]] = None, log_prefix: str = ""):
    """
    Generates content using the Gemini API with function calling for structured output.
    Optionally validates and cleans message IDs if a valid_ids_set is provided.
    """

    try:
        response = gemini_client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        parsed_data = response.parsed
        if parsed_data is not None:
            # logging.info(f"{log_prefix}Model response parsed successfully with {config.response_schema} from {config.response_schema.__class__} content pieces.")
            # If a set of valid IDs is provided, perform validation and self-healing.
            if  valid_ids_set and hasattr(parsed_data, 'validate_and_clean_threads'):
                parsed_data.validate_and_clean_threads(valid_ids_set, log_prefix=log_prefix)

        return parsed_data
    except (IndexError, AttributeError, KeyError) as e:
        # Handle cases where the model doesn't return the expected function call
        logging.error(f"Error parsing model response: {str(e)}")
        empty_data = {field: [] for field, field_info in config.response_schema.model_fields.items() if 'List' in str(field_info.annotation)}
        return config.response_schema.model_validate(empty_data)
