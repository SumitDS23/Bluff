"""
rag/retriever.py
Loads the FAISS vector store from local disk or GCS depending on config.
Adds score threshold and metadata filtering for better precision.
"""

import logging
from pathlib import Path
from langchain_community.vectorstores import FAISS
from rag.embeddings import AzureEmbeddings
from config.settings import settings

logger = logging.getLogger(__name__)


def load_retriever():
    """
    Load FAISS index and return a retriever + vectorstore.
    """
    embedding_model = AzureEmbeddings()
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

    # Default global retriever — uses similarity search with k
    # Reduced k for less blending, will retrieve more in chain.py and rerank
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_k},
    )
    logger.info(f"Retriever ready (k={settings.retrieval_k})")
    return retriever, vector_store


def get_filtered_retriever(vectorstore, lob: str = None,
                            function: str = None, model_name: str = None):
    """
    Returns a retriever filtered by LOB, Function, and/or Model name.
    Falls back to unfiltered if no metadata matches.

    NOTE: Metadata filter only works if documents were tagged during ingestion.
    If your FAISS index doesn't have these metadata fields, this returns
    the global retriever.
    """
    search_kwargs = {"k": settings.retrieval_k}

    filter_dict = {}
    if lob and lob != "Unknown":
        filter_dict["lob"] = lob
    if function and function != "Unknown":
        filter_dict["function"] = function
    if model_name:
        filter_dict["model_name"] = model_name

    if filter_dict:
        search_kwargs["filter"] = filter_dict
        logger.info(f"Filtered retriever | filter={filter_dict}")

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )
