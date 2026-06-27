'''
Input validation + Sanitization layer. Runs before any LLM call, checks for
1. PII (credit card, SSN) - block 
2. Prompt injection
3. White space normalization
'''

import re
from src.agents.state import AgentState

# credit card number
__CC_PATTERN = re.compile(
    r"\b(?:\d[-]?){13,16}\b"
)

# ssn
__SSN_PATTERN = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b"
)

#prompt injection (gemini-2.5-pro generated)
__INJECTION_PHRASES = [
    r"ignore (?:all )?previous instructions",
    r"disregard (?:all )?prior prompts",
    r"system prompt override",
    r"you are now a(?:\s+\w+){1,3}\s+mode",  
    r"act as a",                            
    r"dan mode",                           
    r"forget what we discussed",
    r"new instructions:",
    r"bypass safety filters",
    r"print the system prompt",
    r"reveal your instructions"
]

__INJECTION_RE= re.compile(
    "|".join(__INJECTION_PHRASES),
    re.IGNORECASE
)

def check_input(state:AgentState) -> dict:
    ''' reads state["query"] writes: state for santized query, guardrail flags, error, final response'''
    query:str = state["query"]
    flags:list[str] = list(state.get("guardrail_flags")or [])

    if __CC_PATTERN.search(query) or __SSN_PATTERN.search(query):
        flags.append("INPUT_PII_DETECTED")
        return{
            "santized_query":"",
            "guardrail_flags": flags,
            "error": "Input blocked: PII detected in query",
            "final_response":(
                """
                I am not able to process queries that contains sensitive personal information
                such as credit card or ssn. Please remove the information and try again.
                """
            ),
        }
    
    if __INJECTION_RE.search(query):
        flags.append("INPUT_INJECTION_DETECTED")
        return{
            "santized_query":"",
            "guardrail_flags": flags,
            "error": "Input blocked: Prompt injection in query",
            "final_response":(
                """
                 I am not able to procedd the query, please try again
                """
            ),
        }
    
    sanitized:str = "".join(query.split())
    return{
        "sanitized_query":sanitized,
        "guardrail_flags":flags
    }