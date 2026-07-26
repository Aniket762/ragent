'''
tools exposed:
1. search_policies(query) - semantic search over the vector store
2. list_policy_documents() - lists all ingested policy document names
'''
import asyncio
import logging

import mcp.types as types
import mcp.server as Server
from mcp.server.stdio import stdio_server

from src.config.settings import settings
from src.rag.retriver import build_retriever
from src.rag.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

server = Server("ragent")

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return[
        types.Tool(
            name="search_policies",
            description="return the most relevent policy documents",
            inputSchema={
                "type":"object",
                "properties":{
                    "query":{
                        "type":"string",
                        "description":(
                            "the policy question or topic to search for"
                            "be specific - example. 'credit card point reimbursement limit'"
                            "rather than just 'expenses'"
                        )
                    }
                },
                "required":["query"],
            }
        ),
        types.Tool(
            name="list_policy_documents",
            description=(
                "lists all policy documents that have been ingested and available for search"
                "use this tool when user asks 'what policies do you have?' or"
                "what topics can you help with?"
            ),
            inputSchema={
                "type":"object",
                "properties": {},
                "required":[]
            }
        )
    ]

@server.call_tool()
async def call_tool(name:str, arguments:dict) -> list[types.TextContent]:
    '''
    llm calls this with tool name and args determined from conversation. Run
    appropiate rag operation and return the results as a list of TextContent blocks.

    llm recieves all the text blocks, synthesizes the answer. tool result is context, not final response
    '''
    if name == "search_policies":
        return await _search_policies(arguments.get("query",""))

    elif name=="list_policy_documents":
        return await _list_policy_documents()

    else:
        return [types.TextContent(
            type="text",
            text=f"Unkown tool: {name}"
        )]

async def _search_policies(query:str) -> list[types.TextContent]:
    # direct retrieval call, not master agent call
    if not query.strip():
        return [types.TextContent(
            type="text",
            text="Please provide a search query"
        )]

    try:
        # retriever is synchronous, asyncio.to_thread wraps it in a thread pool
        retriever = build_retriever()
        docs = await asyncio.to_thread(retriever.invoke, query)

        if not docs:
            return [types.TextContent(
                type="text",
                text=(
                    "no relevent docs found for this query"
                )
            )]

        # format each retrieved chunk as seperate textContext blocks
        results:list[types.TextContent]= []
        for i,doc in enumerate(docs,1):
            policy = doc.metadata.get("policy_name","Unkown Policy")
            section = doc.metadata.get("section","")
            header = f"[{i}]{policy}" + (f"- {section}" if section else "")
            results.append(types.TextContent(
                type="text",
                text=f"{header}\n\n{doc.page_content}"
            ))

        return results

    except Exception as exec:
        logger.exception("search_policies tool failed for query:%s",query)
        return [types.TextContent(
            type="text",
            text=f"Search failed: {exec}. policies might not have been ingested"
        )]

async def _list_policy_documents() -> list[types.TextContent]:
    '''
    return a list of distinct policy document names from vector store

    queries chromaDB for all stored document metdata and extracts unique 
    policy_name values, this tells users what topics are available
    '''
    try:
        def _get_names():
            store = get_vectorstore()
            result = store.get(include=["metadatas"])
            metadatas = result.get("metadatas") or []
            names = sorted({
                metadata.get("policy_name","Unkown")
                for metadata in metadatas
                if metadata #skip none
            })
            return names

        names = await asyncio.to_thread(_get_names)

        if not names:
            return [types.TextContent(
                type="text",
                text=(
                    "no policy documents have been ingested yet"
                )
            )]

        doc_list = "\n".join(f" - {name}" for name in names)
        return [types.TextContent(
            type="text",
            text=f"available policy documents ({len(names)}) total:\n\n{doc_list}"
        )]
        
    except Exception as exec:
        logger.exception("list_policy_documents tool failed")
        return[
            types.TextContent(
                type="text",
                text=f"could not retrieve document list:{exec}"
            )
        ]

async def main():
    '''
    start mcp server using stdio transport

    stdio_server() sets up stdin/stdout communication channel that mcp clients
    use to send and receive msg. server runs until client disconnects (stdin closes)
    '''
    logging.basicConfig(level=logging.INFO)
    logger.info(
        "policy agent mcp server starting. "
        "vector store: %s | Embeddings: %s %s",
        settings.chroma_presist_dir,
        settings.embedding_provider,
        settings.embedding_model
    )

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())