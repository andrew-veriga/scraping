# log probability is working in Vertex AI only for Gemini 2.5 models and requires setting  in env vars: 
# GOOGLE_APPLICATION_CREDENTIALS=<your application default credentials json file>
# GOOGLE_CLOUD_REGION=<your cloud region>
# GOOGLE_CLOUD_PROJECT=<your gcp project id>

import os
import logging
from xmlrpc import client

import pandas as pd
from google import genai
from google.genai.types import GenerateContentConfig, GenerateContentResponse
from dotenv import load_dotenv

load_dotenv() # This loads variables from .env into the environment
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# these env vars are set in .env file for local dev
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION")
# USE_VERTEX_AI = os.environ.get("USE_VERTEX_AI", "False").lower() in ("true", "1", "t")
# credentials=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
def check_genai_models():
    for model_id in [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash-001",
        "gemini-2.0-flash-lite-001",
    ]:
        try:
            client = genai.Client(
                vertexai=True, 
                project=PROJECT_ID,
                location=LOCATION,
                # api_key=GEMINI_API_KEY
            )
            response: GenerateContentResponse = client.models.generate_content(
                model=model_id,
                contents="Где находится нофелет?",
                config=GenerateContentConfig(response_logprobs=True, logprobs=5),
            )
        except Exception as e:
            print(f"Error with model {model_id}: {e}")
            continue
        else:
            print(f"Processing model: {model_id}")
            if response.candidates[0].avg_logprobs:
                avg_logprobs = response.candidates[0].avg_logprobs
                num_logprobs = len(response.candidates[0].logprobs_result.chosen_candidates)
                yield dict(
                    model=model_id,
                    response=response.candidates[0].content.parts[0].text,
                    logprobs=True,
                    avg_logprobs=avg_logprobs,
                    num_logprobs=num_logprobs,
                )
            else:
                yield dict(
                    model=model_id,
                    response=response.candidates[0].content.parts[0].text,
                    logprobs=False,
                    avg_logprobs=None,
                    num_logprobs=None,
                )
            
def main():
    # from google.oauth2.service_account import Credentials
    # cr = json.load(open(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")))
    # credentials = Credentials.from_service_account_info(cr)
    # client = genai.Client(
    #             vertexai=True, 
    #             # credentials=credentials,
    #             project=PROJECT_ID,
    #             location=LOCATION,
    #             # api_key=GEMINI_API_KEY
    #         )
    # response: GenerateContentResponse = client.models.generate_content(
    #     model="gemini-2.5-flash",
    #     contents="What's the largest planet in our solar system?",
    #     config=GenerateContentConfig(response_logprobs=True, logprobs=1),
    # )
    # print (response)
    for result in check_genai_models():
        print(result['model'], ":", result['response'])
        print(result['avg_logprobs'])
        print('---'*20)


if __name__ == "__main__":
    main()