from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.config.settings import settings
from src.rag.document_processor import load_and_split, load_directory
from src.rag.vectorstore import add_documents,get_vectorstore

app = typer.Typer(
    name="ingest",
    help="ingest policy documents into chromaDB vector store",
    add_completion=False
)

console = Console()

_POLICIES_DIR = Path("data/policies") # for now add policies here, later sync with S3/Frozen as per usecase

@app.command()
def ingest(
    file: Path | None = typer.Option(
        None, 
        "--file","-f",
        help="ingest a single pdf or docx",
        exists=True,
        file_okay=True,
    ),
    directory: Path = typer.Option(
        _POLICIES_DIR,
        "--dir","-d",
        help="Directory to ingest all docs",
    ),
    clear: bool = typer.Option(
        False,
        "--clear",
        help="Delete all existing docs from vector store before ingesting",
    ),
    dry_run:bool = typer.Option(
        False,
        "--dry-run",
        help="show what would be ingested without writing to vector store"
    )  
) -> None:
    # load docs, split into chunks, embed, store in chromaDB
    if file is not None:
        console.print(f"target single file: {file}")
    else:
        if not directory.exists():
            console.print(f"directory does not exists:{directory}")
            raise typer.Exit(code=1)
        console.print(f"directory->{directory}")

    if clear and not dry_run:
        console.print("clearing vector store ...")
        store = get_vectorstore()
        store._client.delete_collection(settings.chroma_collection_name)

    console.print("loading and chunking docs....")

    # load and split docs
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        task = progress.add_task("Processing ...", total=None)

        if file is not None:
            progress.update(task, description=f"loading {file.name}")
            chunks = load_and_split(file)
        else:
            progress.update(task, description=f"loading all docs in {directory}")
            chunks = load_directory(directory)

    if not chunks:
        console.print("no docs found")
        raise typer.Exit(code=0)

    # extract unique sources
    sources = {chunk.metadata.get("source","unkown") for chunk in chunks}

    if dry_run:   
        console.print("dry run mode no db writes")
        n_written=0
    else:
         with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            progress.add_task("Writing to chromaDB....", total=None)
            n_written = add_documents(chunks)

    # summary table
    table = Table(title="Ingestion Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Source Documents Found", str(len(sources)))
    table.add_row("Total Chunks Created", str(len(chunks)))
    table.add_row("Chunks Stored in DB", str(n_written) if not dry_run else "0 Dry Run")
    table.add_row("Status", "[bold green]Success[/bold green]" if not dry_run else "Dry Run Complete")

    console.print("\n", table)
    
if __name__ == "__main__":
    app()