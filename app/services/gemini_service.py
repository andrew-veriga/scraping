import os
import google.generativeai as genai
from app.models.pydantic_models import RawThreadList, ModifiedRawThreadList, ThreadList, TechnicalTopics, RevisedList

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
model_name = "gemini-2.5-flash"
model = genai.GenerativeModel(model_name)

system_prompt = """
You are an experienced technical support specialist. Your goal is to select solutions from the support chat logs for the FAQ section of your portal dedicated to the Qui blockchain.
"""
prompt_group_logic="""
a) The first message in a thread (topic) ask a question (contains \"?\" or a problem description), describes a problem or expresses a wish;
b) the field Referenced Message ID define the message as a reply to referenced message. The reply cannot be a the first message in the thread (topic) and should continue the existing thread;
c) consider subsequent messages from the same author for relatively short time interval with empty field Referenced Message ID  as a continuation of the conversation if they likely are the addition clarifications;
d) consider messages from the same author repeating the same question as a reminder and a follow-up of the conversation;
e) for short time interval consider consecutive messages of different authors with empty field Referenced Message ID as a part of the same conversation, if they have a common meaning;
f) consider the message as a follow-up for the conversation if the message contains a tag, defined like '<@(\d+)>', where after Commertial At follows Author ID, that author asked the question in the conversation and this message seems to be an answer to the author's question;
"""

prompt_start_step_1 = f"""Analyze the CSV table with messages from the  Discord Channel 'Sui Blockchain Support'.
Your task is group all messages into threads following the logic:
{prompt_group_logic}
Assign a unique identifier to each thread equal to the Message ID of the first message in this thread (topic).
Output the information in the list of JSON objects.
"""

prompt_addition_step1 = f""" 
Analyze the CSV table with messages from the  Discord Channel 'Sui Blockchain Support'.
Your task is group all messages into threads following the Gatheing Logic:
{prompt_group_logic}
Add new threads from CSV following the Gatheing Logic. Assign a unique identifier to each thread equal to the Message ID of the first message in this thread (topic).
Also you already have partially filled threads from previous days:
{{JSON_prev}}
The previous threads was grupped early by the same Gatheing Logic.
Your next task is:
find messages from csv that have 'Referenced Message ID' and non yet gathered in the new threads,
find corresponding Message ID in message lists of previous threads from JSON and add found messages in there as a continue of conversations;
Find other messages that fit mentioned Gatheing Logic. Each message must belong to only one topic.
Finally prepare output:
Only threads fully gathered from CSV table should get status 'new'.
Only threads contained both previous messages from json data and new messages from CSV table should get status 'modified'.
Output only threads with the status of 'new' or 'modified' to the JSON object list.
"""

prompt_addition_step2 = f"""Analyze the CSV table with messages from the  Discord Channel 'Sui Blockchain Support'.
Your task is group all messages into threads following the Gatheing Logic:
{prompt_group_logic}
Output the information in the list of JSON objects.
"""

prompt_step_2 = """
Define a list of keywords and phrases that are typical for technical issues in the context of the Sui blockchain. The list should include:
error, fail, issue, problem, bug, can't, doesn't work, transaction, wallet, rpc, node, gas, smart contract, bridge, stake, unstake, SDK, API, swap, zklogin.
Identify a technical discussions, determine their content and present the results in a structured way.
Analyze the  message of each thread (topic).
Categorize a thread as a \"technical topic\" if its messages contains one or more keywords from the list and it is a question (contains \"?\" or a problem description) or describes a problem.
Ignore irrelevant topics: spam, flood and discussions not directly related to the technical aspects of Sui.
Identify and filter technical topics.
Output the list of 'Topic ID' for technical threads only.
"""

prompt_step_3 = """
Find a solution for each technical topic if it possible.
In each thread find a message that is a solution or answer to the problem or question posed.
As a rule, consider the solution message to be the message from another user (not the topic author) that contains an explanation, instruction or recommendation.
If there are messages with thanks or other confirmations of the verified solution, label it as 'resolved'.
If there are no any confirmation or a solution message folowed by additional unanswered questions, label it as 'suggestion'.
If there are no answers from other users in the thread, or the anwser are irrelevant, label it as 'unresolved'.
If the answer suggests to see other resources, label it as 'outside'.
Derive a general description of the problem and a general description of the solution to the problem if exists.
Actual timestamp of the topic is the datetime of the last message in the thread.
Output the information in the list of JSON objects."""

revision_prompt = """
You have a list of two versions of problem statement and solution with its status:
{pairs}
Your task is to evaluate the improvement of each solution.
If the versions of two problem statements are similar, and the new version of the solution is significantly improved over the previous version, mark them as \"improved\".
If the new version of the problem statement is significantly different from the previous version, label them as \"changed\".
If both of the problem statement and the solution have not got significant changes in the new version, label them as \"persisted\".
"""

generation_config = genai.GenerationConfig(
    temperature=1.0,
)

class _MockParsedResponse:
    """A helper class to mimic the `response.parsed` attribute from the old SDK."""
    def __init__(self, parsed_data):
        self.parsed = parsed_data

def generate_content(contents, schema, config):
    """
    Generates content using the Gemini API with function calling for structured output.

    This function adapts the new google-generativeai SDK to the calling convention
    of the older SDK, expecting a schema and returning an object with a `.parsed` attribute.
    """
    try:
        response = model.generate_content(
            contents=contents,
            generation_config=config,
            tools=[schema]  # Use the Pydantic model for function calling
        )

        # The SDK automatically adds a `from_function_call` method to Pydantic models
        # when they are used as tools.
        tool_call = response.candidates[0].content.parts[0].function_call
        parsed_data = schema.from_function_call(tool_call)
        return _MockParsedResponse(parsed_data)
    except (IndexError, AttributeError, KeyError) as e:
        # Handle cases where the model doesn't return the expected function call
        print(f"Error parsing model response: {e}")
        print(f"Full response: {response}")
        # To prevent a crash, return a mock response with an empty schema instance.
        # This is crucial for list-based fields to avoid TypeErrors in calling code.
        empty_data = {field: [] for field, field_info in schema.model_fields.items() if 'List' in str(field_info.annotation)}
        return _MockParsedResponse(schema.model_validate(empty_data))
