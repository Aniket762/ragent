'''
Using Langsmith for tracing. Traces:
1. LangGraph node execution
2. LLM call
3. Every Retrival with latency, token count, inputs and outputs
'''

import os
from src.config.settings import settings

def setup_langsmith()->bool:
    if not settings.langsmith_enabled:
        return False
    
    os.environ["LANGCHAIN_TRACING_V2"]="true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANCHAIN_PROJECT"] = settings.langsmith_project

    return True

def get_run_url(run_id:str)->str:
    '''
    Build langSmith UI URL for a specific run id
    '''
    project = settings.langsmith_project
    return f"https://smith.langchain.com/o/ragent/projects/p/{project}/r/{run_id}"