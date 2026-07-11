# config file

ROOT = # configure path
path = {
    'cache_dir': ROOT/'cache'/'llm_scp',
    'save_dir': ROOT/'save',
    'output_path': 'ROOT/'llm_scp'/'Generation_PROMPT_MODEL', # CHANGE FOR EACH MODEL !!
    'instance_dir' : ROOT/'llm_scp'/'dataset_instance', 
    'instruction_dir' : ROOT/'llm_scp'/'Prompt'/'PROMPT',
    'example_dir' : ROOT/'llm_scp'/'LLM_example_final',
    'example_match_dir':  ROOT/'llm_scp'/'LLM_example_match_final'
    
}

hp_balanced = {
    'temp_value': 0,
    'top_p_value':0.92,
    'top_k_value':50,
    'repet_value':1.2,
    'sample_bin':True
}
