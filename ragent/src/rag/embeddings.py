from functools import lru_cache
from langchain_core.embeddings import Embeddings
from src.config.settings import settings

@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    '''
    returns a cahced embedding model
    defaults to local sentence transformers model
    '''
    if settings.embedding_provider == "openai":
        if not settings.open_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai"
            )
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.open_api_key
        )
    
    # fallback to local sentence transformer on CPU
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings

    return HuggingFaceBgeEmbeddings(
        model_name= settings.embedding_model,
        model_kwargs = {"device":"cpu"},
        encode_kwargs = {"normalize_embeddings": True}
    )