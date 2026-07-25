'''
ragas for retrieval quality check
context_recall: only for offline evals with known correct answers
'''
from __future__ import annotations
from src.config.settings import settings
from langchain_anthropic import ChatAnthropic

def run_ragas(
        question: str,
        answer:str,
        contexts: list[str],
        ground_truth: str | None = None,
)->dict[str,float]:

    # specify evaluator model
    evaluator_llm = ChatAnthropic(
        model=settings.llm_model,
        anthropic_api_key=settings.anthropic_api_key
    )
    # run ragas on a single triple: question, answer, context
    # defining fn with resource contraint
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evalute
        from ragas.metrics import(
            _answer_relevance,
            _context_precision,
            _faithfulness
        )
    except ImportError as exec:
        raise ImportError("ragas not installed")
    
    # each row is one eval sample
    data:dict[str,list] = {
        "question":[question],
        "answer" : [answer],
        "contexts": [contexts]
    }

    # note: context_recall requires a reference answer
    metrices = [_faithfulness,_answer_relevance,_context_precision]
    if ground_truth is not None:
        from ragas.metrics import _context_recall
        data["ground_truth"] = [ground_truth]
        metrices.append(_context_recall)

    dataset = Dataset.from_dict(data)

    # ragas_eval returns result obj, .to_pandas converts to dataFrame
    result = ragas_evalute(dataset=dataset,metrics=metrices, evaluator_llm=evaluator_llm) # llm calls for eval happens here
    scores_df = result.to_pandas()

    # extract first row as a plain dict, dropping nan
    row = scores_df.iloc[0].dropna().to_dict()

    # keep only metric cols
    metric_names = {m.name for m in metrices}
    return {k: float(v) for k,v in row.items() if k in metric_names}

def run_batch_ragas(samples:list[dict]) -> list[dict[str,float]]:
    return [run_ragas(**sample) for sample in samples] # **sample breaks down the key-val pairs for param matching