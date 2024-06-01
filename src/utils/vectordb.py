import chromadb
from llama_index.core import ServiceContext
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chromadb import ChromaDBVectorStore


def get_chromadb_vector_store(
    client: chromadb.Client, collection_name: str
) -> ChromaDBVectorStore:
    vector_store = ChromaDBVectorStore(
        chroma_client=client, collection_name=collection_name
    )
    return vector_store


def get_index_from_chromadb_vector_store(
    vector_store: ChromaDBVectorStore, service_context: ServiceContext
) -> VectorStoreIndex:
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, service_context=service_context
    )
    return index


def refresh_chromadb_index(documents: list, index: VectorStoreIndex) -> bool:
    try:
        index.refresh_ref_docs(documents)
        status = True
    except Exception:
        status = False
    finally:
        return status
