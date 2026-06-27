from pathlib import Path 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.documents import Document

from src.config.settings import settings

_SUPPORTED_EXTENSIONS = {".pdf",".docx"}

def _enrich_metadata(doc:Document,file_path:Path)->None:
    """
    Inject policy metadata derived from file path
    """
    stem = file_path.stem
    doc.metadata.setdefault("policy_name",stem.split("_")[0] if "_" in stem else stem)
    doc.metadata.setdefault("source",str(file_path))
    doc.metadata.setdefault("file_type",file_path.suffix.lstrip("."))

def load_and_split(file_path: str| Path) -> list[Document]:
    """
    load a single pdf/docx and return metadata-enriched chunks
    """
    path = Path(file_path)
    if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    
    loader = UnstructuredFileLoader(str(path),mode="elements")
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = settings.chunk_size,
        chunk_overlap = settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    for chunk in chunks:
        _enrich_metadata(chunk,path)
    return chunks

def load_directory(directory: str | Path) -> list[Document]:
    """
    load ALL pdfs and docx found recursively in the given path
    """
    dir_path = Path(directory)
    all_chunks:list[Document] = []

    for ext in _SUPPORTED_EXTENSIONS:
        for file_path in dir_path.glob(f"**/*{ext}"):
            all_chunks.extend(load_and_split(file_path))
    
    return all_chunks