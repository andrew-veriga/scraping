import yaml
import os

_prompts = {}

def load_prompts():
    global _prompts
    # Construct the absolute path to the prompts.yaml file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    prompts_path = os.path.join(dir_path, '..', '..', 'configs', 'prompts.yaml')

    with open(prompts_path, 'r') as f:
        _prompts = yaml.safe_load(f)

    # Dynamically assign prompts to module-level variables
    for key, value in _prompts.items():
        globals()[key] = value

def reload_prompts():
    load_prompts()

# Initial load of the prompts
load_prompts()

# You can still define f-string prompts here if they need dynamic values from other modules,
# but it's generally better to handle formatting when the prompt is used.

# These will be formatted when used, not here

prompt_addition_step1 = _prompts.get('prompt_addition_step1', '')  

revision_prompt = _prompts.get('revision_prompt', '')