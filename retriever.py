"""
rag/retriever.py
Loads FAISS vector stores from local disk.

Changes vs previous version:
- load_retriever() now accepts a store parameter: "ml" | "general"
- Reads from settings.local_index_path_ml or settings.local_index_path_general
- Old single-store path (settings.local_index_path) kept as fallback alias
  so existing code that hasn't migrated yet doesn't break
- get_filtered_retriever() unchanged
"""

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS

from rag.embeddings import AzureEmbeddings
from config.settings import settings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Store loader — parameterised
# ──────────────────────────────────────────────────────────────────────────────

def load_retriever(store: str = "ml"):
    """
    Load a FAISS retriever for the specified store.

    Parameters
    ----------
    store : "ml" | "general"
        "ml"      → faiss_index_ml      (model_doc, cde, framework, general)
        "general" → faiss_index_general (newsletter, utility)

    Returns
    -------
    (retriever, vectorstore)
        Same tuple shape as before so main.py callers need minimal changes.
    """
    embedding_model = AzureEmbeddings()

    # Resolve the correct index path
    if store == "general":
        local_path = settings.local_index_path_general
        store_label = "General"
    else:
        # Default / "ml" — also handles legacy calls with no argument
        local_path = settings.local_index_path_ml
        store_label = "ML"

    # Validate path exists
    if not Path(local_path).exists():
        raise FileNotFoundError(
            f"FAISS [{store_label}] index not found at '{local_path}'. "
            f"Run ingest/ingest.py first."
        )

    logger.info(f"Loading FAISS [{store_label}] index from: {local_path}")

    vector_store = FAISS.load_local(
        local_path,
        embedding_model,
        allow_dangerous_deserialization=True,
    )

    retriever = vector_store.as_retriever(
        search_type=settings.retrieval_mode,
        search_kwargs={"k": settings.retrieval_k},
    )

    logger.info(f"Retriever [{store_label}] ready")
    return retriever, vector_store


# ──────────────────────────────────────────────────────────────────────────────
# Filtered retriever — unchanged, used by RAG flow in main.py
# ──────────────────────────────────────────────────────────────────────────────

def get_filtered_retriever(vectorstore, lob: str = None, function: str = None):
    """
    Returns a retriever filtered by LOB and/or Function metadata.
    Falls back to unfiltered search if no metadata filter is applicable.

    Parameters
    ----------
    vectorstore : FAISS vectorstore object (returned from load_retriever)
    lob         : LOB string to filter on, e.g. "ABSLI"
    function    : Function string to filter on, e.g. "HR"
    """
    search_kwargs: dict = {"k": settings.retrieval_k}

    filter_dict: dict = {}
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
