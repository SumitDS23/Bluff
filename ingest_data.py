"""
ingest/ingest.py
Enhanced multi-document-type ingestion pipeline with adaptive chunking.

Supports 4 document types with optimized chunking strategies:
1. Model Documentation → 400-500 token chunks
2. CDE (Critical Data Elements) → 200-300 token chunks
3. Data & Analytics Framework → 600-800 token chunks  
4. Newsletters → 500-700 token chunks

Run with: python ingest/ingest.py

Dependencies:
pip install markitdown[all]
pip install langchain-core langchain-community langchain-text-splitters
pip install langchain-openai faiss-cpu
"""

import sys
import logging
import re
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Document Type Detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_document_type(file_path: Path) -> str:
    """
    Detect document type from folder structure.
    Returns: "model_doc" | "cde" | "framework" | "newsletter" | "general"
    """
    parts = [p.lower() for p in file_path.parts]
    
    # CDE detection
    if "critical data elements" in parts or any("cde" in p for p in parts):
        return "cde"
    
    # Framework detection
    if "data & analytics framework" in parts or any("framework" in p.lower() for p in parts):
        return "framework"
    
    # Newsletter detection
    if "newsletter" in parts:
        return "newsletter"
    
    # Model Documentation (has LOB subfolder structure)
    if "model documentation" in parts:
        return "model_doc"
    
    # General (Standards, MLOps, etc.)
    if "general" in parts or "standards" in parts or "best practices" in parts:
        return "general"
    
    return "model_doc"  # default


# ══════════════════════════════════════════════════════════════════════════════
# Metadata Extractors (per document type)
# ══════════════════════════════════════════════════════════════════════════════

def extract_model_doc_metadata(file_path: Path, parts: list) -> dict:
    """Extract metadata for Model Documentation."""
    docs_index = next(
        (i for i, p in enumerate(parts) if p.lower() in ["knowledgebase_word", "model documentation"]),
        None
    )
    
    # Standard structure: KnowledgeBase_word/Model Documentation/LOB/Function/filename
    lob = "Unknown"
    function = "Unknown"
    
    if docs_index is not None:
        # Try to find LOB (should be after "Model Documentation")
        lob_index = docs_index + 2  # skip "Model Documentation"
        if lob_index < len(parts):
            lob = parts[lob_index].replace("_", " ")
        
        # Function is next level
        func_index = lob_index + 1
        if func_index < len(parts):
            function = parts[func_index].replace("_", " ")
    
    model_name = file_path.stem.replace("_", " ")
    
    return {
        "doc_type": "model_doc",
        "lob": lob,
        "function": function,
        "model_name": model_name,
        "source": str(file_path),
        "filename": file_path.name,
        "file_type": file_path.suffix.lower(),
    }


def extract_cde_metadata(file_path: Path, content: str) -> dict:
    """
    Extract metadata for CDE documents.
    CDE structure: Critical Data Elements/LOB_CDE.xlsx
    """
    filename = file_path.stem
    lob = "Unknown"
    
    # Extract LOB from filename (e.g., "ABSLI_CDE" → "ABSLI")
    match = re.match(r"([A-Z]+)_CDE", filename, re.IGNORECASE)
    if match:
        lob = match.group(1).upper()
    
    return {
        "doc_type": "cde",
        "lob": lob,
        "domain": "data_governance",  # all CDEs are governance docs
        "source": str(file_path),
        "filename": file_path.name,
        "file_type": file_path.suffix.lower(),
    }


def extract_framework_metadata(file_path: Path) -> dict:
    """
    Extract metadata for Data & Analytics Framework documents.
    Structure: Data & Analytics Framework/LOB_Mar'25.xlsx
    """
    filename = file_path.stem
    lob = "Unknown"
    month = "Unknown"
    
    # Extract LOB and month (e.g., "ABSLI_Mar'25" → LOB=ABSLI, month=Mar'25)
    match = re.match(r"([A-Z]+)_(.*)", filename, re.IGNORECASE)
    if match:
        lob = match.group(1).upper()
        month = match.group(2).replace("'", "")
    
    return {
        "doc_type": "framework",
        "lob": lob,
        "period": month,
        "topic": "analytics_framework",
        "source": str(file_path),
        "filename": file_path.name,
        "file_type": file_path.suffix.lower(),
    }


def extract_newsletter_metadata(file_path: Path) -> dict:
    """
    Extract metadata for Newsletter documents.
    Structure: Newsletter/Newsletter_Dec25.docx
    """
    filename = file_path.stem
    period = "Unknown"
    
    # Extract month (e.g., "Newsletter_Dec25" → "Dec25")
    match = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\d{2}", filename, re.IGNORECASE)
    if match:
        period = match.group(0)
    
    return {
        "doc_type": "newsletter",
        "period": period,
        "topic": "updates",
        "source": str(file_path),
        "filename": file_path.name,
        "file_type": file_path.suffix.lower(),
    }


def extract_general_metadata(file_path: Path) -> dict:
    """Extract metadata for General/Standards documents."""
    filename = file_path.stem.replace("_", " ")
    
    # Detect topic from filename
    topic = "standards"
    if "mlops" in filename.lower():
        topic = "mlops"
    elif "model documentation" in filename.lower():
        topic = "model_documentation_standards"
    elif "inventory" in filename.lower():
        topic = "inventory_definitions"
    
    return {
        "doc_type": "general",
        "topic": topic,
        "source": str(file_path),
        "filename": file_path.name,
        "file_type": file_path.suffix.lower(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Document Loading with Type Detection
# ══════════════════════════════════════════════════════════════════════════════

def load_documents(doc_path: str) -> List[Tuple[object, str]]:
    """
    Convert all supported documents to Markdown using MarkItDown.
    Returns list of (Document, doc_type) tuples for adaptive chunking.
    """
    from markitdown import MarkItDown
    from langchain_core.documents import Document

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

    # Track counts per type
    type_counts = {}

    for file_path in files:
        logger.info(f"Converting: {file_path.name}")
        
        try:
            result = md_converter.convert(str(file_path))
            markdown_content = result.text_content

            if not markdown_content or not markdown_content.strip():
                logger.warning(f"  SKIP: Empty content in {file_path.name}")
                continue
            
            # Detect document type
            doc_type = detect_document_type(file_path)
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            
            # Extract metadata based on type
            parts = file_path.parts
            if doc_type == "model_doc":
                metadata = extract_model_doc_metadata(file_path, parts)
            elif doc_type == "cde":
                metadata = extract_cde_metadata(file_path, markdown_content)
            elif doc_type == "framework":
                metadata = extract_framework_metadata(file_path)
            elif doc_type == "newsletter":
                metadata = extract_newsletter_metadata(file_path)
            else:  # general
                metadata = extract_general_metadata(file_path)
            
            doc = Document(
                page_content=markdown_content,
                metadata=metadata
            )
            documents.append((doc, doc_type))
            
            logger.info(
                f"  OK: {file_path.name} | Type={doc_type} | "
                f"{metadata.get('lob', metadata.get('topic', ''))} | "
                f"({len(markdown_content):,} chars)"
            )
        
        except Exception as e:
            logger.error(f"  FAIL: {file_path.name} | Error: {e}")
            raise  # Stop on first error

    logger.info(f"Successfully loaded {len(documents)} document(s)")
    logger.info(f"Document type distribution: {type_counts}")
    return documents


# ══════════════════════════════════════════════════════════════════════════════
# Adaptive Chunking (per document type)
# ══════════════════════════════════════════════════════════════════════════════

def split_documents_adaptive(documents: List[Tuple[object, str]]) -> list:
    """
    Split documents using adaptive chunking based on document type.
    
    Chunk size strategy:
    - model_doc:   400-500 tokens (detailed but focused)
    - cde:         200-300 tokens (field-level granularity)
    - framework:   600-800 tokens (conceptual chunks with context)
    - newsletter:  500-700 tokens (section-based)
    - general:     500 tokens (standard)
    """
    from langchain_text_splitters import (
        MarkdownTextSplitter,
        RecursiveCharacterTextSplitter
    )

    if not documents:
        return []

    all_chunks = []
    
    # Splitter configurations per type
    splitters = {
        "model_doc": MarkdownTextSplitter(
            chunk_size=500,      # Reduced from 1000
            chunk_overlap=50,    # Reduced from 200
        ),
        "cde": RecursiveCharacterTextSplitter(
            chunk_size=300,      # Small for field definitions
            chunk_overlap=30,
            separators=["\n\n", "\n", "|", " ", ""],  # table-friendly
        ),
        "framework": MarkdownTextSplitter(
            chunk_size=700,      # Larger for conceptual content
            chunk_overlap=70,
        ),
        "newsletter": MarkdownTextSplitter(
            chunk_size=600,
            chunk_overlap=60,
        ),
        "general": MarkdownTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        ),
    }
    
    # Group documents by type
    docs_by_type = {}
    for doc, doc_type in documents:
        if doc_type not in docs_by_type:
            docs_by_type[doc_type] = []
        docs_by_type[doc_type].append(doc)
    
    # Split each group with appropriate splitter
    for doc_type, docs in docs_by_type.items():
        splitter = splitters.get(doc_type, splitters["general"])
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)
        logger.info(
            f"  {doc_type.upper()}: {len(docs)} doc(s) → {len(chunks)} chunk(s) "
            f"(size={splitter.chunk_size}, overlap={splitter.chunk_overlap if hasattr(splitter, 'chunk_overlap') else 'N/A'})"
        )
    
    logger.info(f"Total: {len(documents)} doc(s) → {len(all_chunks)} chunk(s)")
    return all_chunks


# ══════════════════════════════════════════════════════════════════════════════
# Index Building
# ══════════════════════════════════════════════════════════════════════════════

def build_index(chunks: list):
    """
    Embed chunks using Azure OpenAI embeddings and save FAISS index to disk.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_openai import AzureOpenAIEmbeddings

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
    
    # Log metadata stats
    log_metadata_stats(chunks)
    
    return vectorstore


def log_metadata_stats(chunks: list):
    """Log metadata statistics for verification."""
    doc_types = {}
    lobs = {}
    
    for chunk in chunks:
        dt = chunk.metadata.get("doc_type", "unknown")
        doc_types[dt] = doc_types.get(dt, 0) + 1
        
        lob = chunk.metadata.get("lob", "N/A")
        if lob != "N/A":
            lobs[lob] = lobs.get(lob, 0) + 1
    
    logger.info("=" * 60)
    logger.info("Metadata Statistics:")
    logger.info(f"  Document types: {doc_types}")
    logger.info(f"  LOB distribution: {lobs}")
    logger.info("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("COE Analytics — Multi-Type Document Ingestion")
    logger.info("=" * 60)

    logger.info(f"Loading documents from: {settings.local_docs_path}")
    documents = load_documents(settings.local_docs_path)

    if not documents:
        logger.error("No documents loaded. Exiting.")
        sys.exit(1)

    logger.info("Splitting documents with adaptive chunking...")
    chunks = split_documents_adaptive(documents)

    if not chunks:
        logger.error("No chunks created. Exiting.")
        sys.exit(1)

    logger.info("Building FAISS index with rich metadata...")
    build_index(chunks)

    logger.info("=" * 60)
    logger.info(f"✅ Ingestion complete!")
    logger.info(f"   {len(documents)} documents → {len(chunks)} chunks")
    logger.info(f"   Index saved to: {settings.local_index_path}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
