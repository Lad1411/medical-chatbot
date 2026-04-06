from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from tqdm import tqdm

def create_keyword_retriever(doc):
    """
        Create Vietnamese keyword retriever
        Args:
            chunks: list of langchain Documents
    """

    retriever = BM25Retriever.from_documents(
        documents=doc,
    )
    retriever.k = 2
    return retriever

def create_vector_db(doc, embedd_model, db_dir='../vector_db'):
    """
        Create Chroma vector DB
        Args:
            chunks: list of langchain Documents
            db_dir: directory of vector db
            embed_model: embedding model
    """

    vector_store = Chroma(
        collection_name='langchain_store',
        embedding_function=embedd_model,
        persist_directory=db_dir,
        collection_metadata={"hnsw:space": "cosine"}
    )

    batch_size = 5000
    total_chunks = len(doc)
    progressbar = tqdm(range(0, total_chunks, batch_size), desc="Ingesting to ChromaDB")

    for i in progressbar:
        batch_chunks = doc[i: (i+batch_size)]
        vector_store.add_documents(batch_chunks)
    return vector_store

