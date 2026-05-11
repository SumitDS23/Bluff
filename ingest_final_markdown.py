"""
ingest/ingest.py
Document ingestion pipeline using MarkItDown for conversion
and MarkdownTextSplitter for chunking.

Supports: .pdf, .docx, .pptx, .xlsx
Preserves table structure via Markdown conversion.

Run with: python ingest/ingest.py

Dependencies:
pip install markitdown[all]
pip install langchain-core langchain-community langchain-text-splitters
pip install langchain-openai faiss-cpu
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from config.settings import settings

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def load_documents(doc_path: str) -> list:
    """
    Convert all supported documents to Markdown using MarkItDown.
    Supports: .pdf, .docx, .pptx, .xlsx
    Returns a list of LangChain Document objects.
    Raises an exception immediately if any file fails to convert.
    """
    from markitdown import MarkItDown
    from langchain_core.documents import Document  # Fixed: was langchain.schema

    documents = []
    md_converter = MarkItDown()
    supported = {".pdf", ".docx", ".pptx", ".xlsx"}

    doc_dir = Path(doc_path)
    if not doc_dir.exists():
        raise FileNotFoundError(
            f"Documents folder not found at '{doc_path}'. "
            "Set LOCAL_DOCS_PATH in your .env file."
        )

    files = [f for f in doc_dir.glob("**/*") if f.suffix.lower() in supported]

    if not files:
        logger.warning(f"No supported documents found in '{doc_path}'")
        return []

    logger.info(f"Found {len(files)} document(s) to ingest")

    for file_path in files:
        logger.info(f"Converting: {file_path.name}")
        # No try/except — will raise and stop immediately on any failure
        result = md_converter.convert(str(file_path))
        markdown_content = result.text_content

        if not markdown_content or not markdown_content.strip():
            raise ValueError(
                f"Empty content after conversion: {file_path.name}. "
                "The file may be corrupted or password-protected."
            )
        
        parts = file_path.parts
        docs_index=next(
            (i for i, p in enumerate(parts) if p.lower() == "knowledgebase_word"),
            None
        )
        logger.info(f"DEBUG | parts={parts} | docs_index={docs_index}")
        lob = (
            parts[docs_index + 1].replace("_", " ")
            if docs_index is not None and len(parts) > docs_index + 1
            else "Unknown"
        )
        function = (
            parts[docs_index + 2].replace("_", " ")
            if docs_index is not None and len(parts) > docs_index + 2
            else "Unknown"
        )
        model_name = file_path.stem.replace("_", " ")
         
        doc = Document(
            page_content=markdown_content,
            metadata={
                "source": str(file_path),
                "filename": file_path.name,
                "type": file_path.suffix.lower(),
                "lob": lob,
                "function": function,
                "model_name": model_name,
            }
        )
        documents.append(doc)
        logger.info(f"  OK: {file_path.name} "
            f"| LOB:{lob} "
            f"| Function: {function} "
            f"|({len(markdown_content):,} chars)")

    logger.info(f"Successfully loaded {len(documents)} document(s)")
    return documents

def split_documents(documents: list) -> list:
    """
    Split documents using MarkdownTextSplitter to preserve
    table structure and heading hierarchy.
    """
    from langchain_text_splitters import MarkdownTextSplitter  # Fixed: was langchain.text_splitter

    if not documents:
        return []

    splitter = MarkdownTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    chunks = splitter.split_documents(documents)
    logger.info(
        f"Split {len(documents)} document(s) into {len(chunks)} chunk(s) "
        f"(chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap})"
    )
    return chunks

def build_index(chunks: list):
    """
    Embed chunks using Azure OpenAI embeddings and save FAISS index to disk.

    Required settings (in .env / config/settings.py):
        AZURE_OPENAI_API_KEY
        AZURE_OPENAI_ENDPOINT
        AZURE_OPENAI_EMBED_DEPLOYMENT   (e.g. "text-embedding-ada-002")
        AZURE_OPENAI_API_VERSION        (e.g. "2024-02-01")
        LOCAL_INDEX_PATH
    """
    from langchain_community.vectorstores import FAISS
    from langchain_openai import AzureOpenAIEmbeddings  # Fixed: replaces GeminiEmbeddings

    if not chunks:
        raise ValueError("No chunks to embed. Check your documents folder.")

    index_path = Path(settings.local_index_path)
    index_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Embedding {len(chunks)} chunks using Azure deployment: "
        f"{settings.azure_embedding_deployment}"
    )

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=settings.azure_embedding_deployment,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )

    # Embed in batches to avoid throttling
    batch_size = 50
    vectorstore = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        logger.info(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)")

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            batch_store = FAISS.from_documents(batch, embeddings)
            vectorstore.merge_from(batch_store)

    vectorstore.save_local(str(index_path))
    logger.info(f"FAISS index saved to: {index_path}")
    return vectorstore

def main():
    logger.info("=" * 60)
    logger.info("COE Analytics — Document Ingestion Pipeline")
    logger.info("=" * 60)

    logger.info(f"Loading documents from: {settings.local_docs_path}")
    documents = load_documents(settings.local_docs_path)

    if not documents:
        logger.error("No documents loaded. Exiting.")
        sys.exit(1)

    logger.info("Splitting documents into chunks...")
    chunks = split_documents(documents)

    if not chunks:
        logger.error("No chunks created. Exiting.")
        sys.exit(1)

    logger.info("Building FAISS index...")
    build_index(chunks)

    logger.info("=" * 60)
    logger.info(f"Ingestion complete! {len(documents)} doc(s) -> {len(chunks)} chunk(s)")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
