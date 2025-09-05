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

prompt_start_step_1 = _prompts.get('prompt_start_step_1', '').format(
    prompt_group_logic=_prompts.get('prompt_group_logic', '')
)

prompt_addition_step1 = _prompts.get('prompt_addition_step1', '').format(
    prompt_group_logic=_prompts.get('prompt_group_logic', '')
)

prompt_addition_step2 = _prompts.get('prompt_addition_step2', '').format(
    prompt_group_logic=_prompts.get('prompt_group_logic', '')
)

revision_prompt = _prompts.get('revision_prompt', '')