'''
judge recieves the og query, retrieved policy chunks and the generated answer. Scores in 4d:
1. faithfulness: is every claim in the answer supported by retrieved docs (0:completely madeup, 1: fully grounded) : wt->0.50
2. relevence: address user's query (0:completely off-topic,1: directly answers): wt->0.30
3. completeness (0: only scratched the surface, 1:thorough): wt->0.20
4. hallucination: (true: hallucination detected, false: no hallucination found) : cap at 0.4
'''
from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage,SystemMessage
from pydantic import BaseModel, Field

from src.config.settings import settings

class EvaluatationResult(BaseModel):
    '''
    structured score returned by judge, [0.0,1.0]
    reasoning is the explanation, only for debugging
    '''
    faithfulness:float = Field(
        ge=0.0,le=1.0,
        description="How well the answer is grounded in the retrieved docs"
    )

    relevance:float = Field(
        ge=0.0,le=1.0,
        description="How directly the answer addresses the user's question"
    )

    completeness:float = Field(
        ge=0.0,le=1.0,
        description="How thoroughly the answer covers the relevant aspects from the docs"
    )

    hallucination:bool = Field(
        description="True if the answer contains specific facts not found in retrived doc"
    )

    reasoning:str = Field(
        description="judge's step by step explanation from scores"
    )

    @property
    def overall_score(self)->float:
        base = (self.faithfulness*0.50 + self.relevance*0.30 + self.completeness*0.20)
        if self.hallucination:
            return min(base,0.40)
        return base

    @property
    def passed(self)->bool:
        return self.overall_score >=0.7 and not self.hallucination
    
@lru_cache(maxsize=1)
def _get_judge_llm():
    'using .with_structured_output ensures we always get a valid json score back, no matter what format the model happens to respond in'
    return ChatAnthropic(
        model=settings.llm_model,
        max_tokens = 1024,
        temperature=0.0,
        anthropic_api_key= settings.anthropic_api_key
    ).with_structured_output(EvaluatationResult)

_JUDGE_SYSTEM_PROMPT="tbd"

def _format_docs_for_judge(docs:list[Document])-> str:
    parts:list[str] = []
    for i, doc in enumerate(docs,1):
        policy = doc.metadata.get("policy_name","Unknown Policy")
        section = doc.metadata.get("section"," ")
        header = f"[{i}]{policy}" + (f" - {section}" if section else "")
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)

def evaluate(
        query:str,
        retrieved_docs: list[Document],
        generated_response: str
) -> EvaluatationResult:
    '''
    use asyncio.gather() to parallelize for batch evaluation
    '''
    context = _format_docs_for_judge(retrieved_docs)
    human_message = f"""\
        Question: {query}
        Retrieved policy excerpts: {context}
        Generated answer: {generated_response}
    """

    result: EvaluatationResult = _get_judge_llm().invoke([
        SystemMessage(content=_JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=human_message)
    ])

    return result


