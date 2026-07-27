from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings,SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # llm
    anthropic_api_key:str = ""
    llm_model:str= "claude-opus-4-8"
    llm_max_tokens:int = 4096
    llm_temperature:float=0.1

    # embeddings
    embedding_provider: Literal["local","openai"] = "local"
    embedding_model:str = "all-MiniLM-L6-v2"
    open_api_key:str = ""

    # vector db
    chroma_presist_dir:str = "./data/chroma-db"
    chroma_collection_name: str ="policies"

    # retrival
    retriver_top_k:int=6
    use_bm25:bool = False

    # chunking
    chunk_size:int=100
    chunk_overlap:int = 200

    # observalibility
    langsmith_api_key:str = ""
    langsmith_project:str = "ragent"
    langsmith_tracing:bool = False

    # tasks
    redis_url: str = "redis://localhost:6317"

    @property
    def langsmith_enabled(self) -> bool:
        return bool(self.langsmith_api_key) and self.langsmith_tracing
    

settings = Settings()