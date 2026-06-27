'''
runs after policy agent generates a response, before the response is returned to the caller
1. PII removal
2. Grounding Heuristic - to check if response contains lots of words not present in the retrived document
'''

import re
from langchain_core.documents import Document
from src.agents.state import AgentState

# credit card number
__CC_PATTERN = re.compile(
    r"\b(?:\d[-]?){13,16}\b"
)

# ssn
__SSN_PATTERN = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b"
)

def _redact_pii(text:str) -> tuple[str,bool]:
    redacted = __CC_PATTERN.sub("[REDACTED-CC]",text)
    redacted = __SSN_PATTERN.sub("[REDACTED-SSN]",text)
    return redacted, redacted!=text

def _is_grounded(response:str, docs:list[Document]) -> bool:
    '''
    1. count words with len >3 from retrived docs
    2. do same for response
    3. if overlap is less make the flag true, of 20 words less than 3 matches

    Will act as a gate for model hallucination
    '''
    if not docs:
        return True
    
    doc_vocab: set[str]=set()
    for doc in docs:
        doc_vocab.update(
            w.lower() for w in doc.page_content.split() if len(w)>4
        )
    
    response_words = [ w.lower() for w in response.split() if len(w)>4]
    response_vocab = set(response_words)

    overlap = len(response_words & doc_vocab)

    # ignoring short responses,like when model returns "I could not find the answer"
    if len(response_words)>20 and overlap<3:
        return False 
    return True

def check_output(state:AgentState) -> dict:
    """
    reads: state["generated_response], state["retrived_docs"]
    writes: state["final_response"], state["guardrail_flags"]

    final_response is what callers display to users - always post guardrail version
    """
    response:str = state.get("generated_response","")
    docs: list[Document] = state.get("retrived_docs",[])
    flags:list[str] = list(state.get("guardrail_flags")or [])

    redacted,was_redacted = _redact_pii(response)
    if was_redacted:
        flags.append("OUTPUT_PII_REDACTED")

    if not _is_grounded(redacted,docs):
        flags.append("OUTPUT_LOW_GROUND_SCORE")
    
    return {
        "final_response":redacted,
        "guardrail_flags":flags
    }
