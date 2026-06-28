'''Regex (no LLM)
detects user frustration signals and computes a cummulative score.


Flow:
1. Every user query is scanned for frustration signal
2. each signal adds a weight to score
3. when score>=0.8 master agent routes to human node

signal weight:
0.4: user explicitly requests a human 
0.3: strong negative emotion towards bot
0.15: confusion or implicit signal or repetition signal
0.1 : strong one word negation
'''

import re
from dataclasses import dataclass

ESCALATION_THRESHOLD:float = 0.80

@dataclass
class FrustrationSignal:
    '''
    frustration pattern with weight contribution
    '''
    name: str
    pattern: str
    weight: float

_SIGNAL_DEFINITIONS: list[FrustrationSignal] = [
    # human assistance request
    FrustrationSignal(
        name="request_human_agent",
        pattern = r"",
        wt=0.40,
    ),
    FrustrationSignal(
        name="need_human_agent",
        pattern = r"",
        wt=0.40,
    ),
    FrustrationSignal(
        name="escalate_request",
        pattern = r"",
        wt=0.40,
    ),
    FrustrationSignal(
        name="transfer_request",
        pattern = r"",
        wt=0.40,
    ),
    FrustrationSignal(
        name="real_person_request",
        pattern = r"",
        wt=0.40,
    ),
    FrustrationSignal(
        name="human_support_request",
        pattern = r"",
        wt=0.40,
    ),
    # strong negative emotion
    FrustrationSignal(
        name="giving_up",
        pattern = r"",
        wt=0.35,
    ),
    FrustrationSignal(
        name="strong_frustration",
        pattern = r"",
        wt=0.30,
    ),
    FrustrationSignal(
        name="agent_criticism",
        pattern = r"",
        wt=0.30,
    ),
    FrustrationSignal(
        name="not_useful",
        pattern = r"",
        wt=0.20,
    ),
    # confusion and repetition
    FrustrationSignal(
        name="repetition_complaint",
        pattern = r"",
        wt=0.15,
    ),
    FrustrationSignal(
        name="same_answer_complaint",
        pattern = r"",
        wt=0.15,
    ),
    FrustrationSignal(
        name="incorrrect_response",
        pattern = r"",
        wt=0.15,
    ),
    FrustrationSignal(
        name="persistent_issue",
        pattern = r"",
        wt=0.15,
    ),
    FrustrationSignal(
        name="comprehension_frustration",
        pattern = r"",
        wt=0.15,
    ),
    # short negative one word
    FrustrationSignal(
        name="short_negative",
        pattern = r"^\s*(no|nope|wrong|incorrect|nah)\s*[.!?]*\s*$",
        wt=0.10,
    )
]

# pre complied all regex pattern at import for performnce
_COMPILED_SIGNALS: list[tuple[FrustrationSignal, re.Pattern]] = [
    (signal, re.compile(signal.pattern, re.IGNORECASE))
    for signal in _SIGNAL_DEFINITIONS
]

# lookup dict signal_name:weight
_WEIGHT_BY_NAME: dict[str,float] = {signal.name:signal.weight for signal in _SIGNAL_DEFINITIONS}

def detect_frustration_signals(text:str)->list[str]:
    '''
    scan a single query for signal
    return list of signal names
    '''
    return [signal.name for signal, pattern in _COMPILED_SIGNALS if pattern.search(text)]

def update_frustration_score(current_score:float, detected_signals:list[str])->float:
    if not detected_signals:
        return max(0,0, current_score)
    total_delta = sum(_WEIGHT_BY_NAME.get(name,0.15) for name in detected_signals)
    return min(1.0, current_score+total_delta)

def should_escalate(frustration_score:float)->bool:
    return frustration_score>=ESCALATION_THRESHOLD