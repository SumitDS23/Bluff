"""
rag/retriever.py
Loads the FAISS vector store from local disk or GCS depending on config.
"""

import logging
from pathlib import Path
from langchain_community.vectorstores import FAISS
from rag.embeddings import AzureEmbeddings          # ← Changed
from config.settings import settings

logger = logging.getLogger(__name__)


def load_retriever():
    """
    Load FAISS index and return a retriever.
    - If USE_GCS=true: downloads index from GCS first
    - Otherwise: loads from LOCAL_INDEX_PATH
    """
    embedding_model = AzureEmbeddings()              # ← Changed
    local_path = settings.local_index_path

    if settings.use_gcs:
        logger.info("USE_GCS=true — downloading index from GCS...")
        from storage.gcs import download_index
        success = download_index(
            bucket_name=settings.gcs_bucket,
            gcs_prefix=settings.gcs_index_prefix,
            local_path=local_path,
        )
        if not success:
            raise FileNotFoundError(
                f"FAISS index not found in GCS bucket '{settings.gcs_bucket}' "
                f"at prefix '{settings.gcs_index_prefix}'. Run ingest first."
            )
    else:
        if not Path(local_path).exists():
            raise FileNotFoundError(
                f"FAISS index not found at '{local_path}'. Run ingest first."
            )

    logger.info(f"Loading FAISS index from {local_path}...")
    vector_store = FAISS.load_local(
        local_path, embedding_model,
        allow_dangerous_deserialization=True,
    )

    retriever = vector_store.as_retriever(
        search_type=settings.retrieval_mode,
        search_kwargs={"k": settings.retrieval_k},
    )
    logger.info("Retriever ready")
    return retriever, vector_store

# def _load_vectorstore():
    # """Load FAISS vectorstore from disk."""
    
    # embeddings = AzureEmbeddings()
    
    # return FAISS.load_local(
        # settings.local_index_path,
        # embeddings,
        # allow_dangerous_deserialization=True,
    # )
    
def get_filtered_retriever(vectorstore, lob: str = None, function: str = None):
    """
    Returns a retriever filtered by LOB and/or Function.
    Falls back to unfiltered if no metadata matches.
    """  
    search_kwargs = {"k": settings.retrieval_k}  
    # Build metadata filter
    filter_dict = {}
    if lob and lob != "Unknown":
        filter_dict["lob"] = lob
    if function and function != "Unknown":
        filter_dict["function"] = function
    
    if filter_dict:
        search_kwargs["filter"] = filter_dict
    
    return vectorstore.as_retriever(
        search_type=settings.retrieval_mode,
        search_kwargs=search_kwargs,
    )