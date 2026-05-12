"""
rag/chain.py
Builds the RAG chain: retriever -> prompt -> LLM -> response.
Uses Azure OpenAI + LLM-based query rewriting + cross-encoder reranking.
"""

import logging
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from config.settings import settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Cross-encoder reranker — loaded once at module level
# ──────────────────────────────────────────────────────────────────────────
_reranker = None

def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512,
            )
            logger.info("Cross-encoder reranker loaded")
        except Exception as e:
            logger.warning(f"Reranker not available: {e}")
            _reranker = False
    return _reranker if _reranker else None


SYSTEM_PROMPT =  """You are a senior Analytics and MLOps expert assisting internal analytics teams with Model Inventory, Model Documentation, and Model Governance queries.
You must answer strictly and only using the retrieved context provided in the conversation.

The retrieved context may contain:
- Structured inventory data (model lists, description, LOB, owner, status, type, etc.)
- Narrative model documentation (objective, methodology, features, algorithms, performance)
- Standards & Best Practices documents (MLOps framework, governance, monitoring)


ADDITIONAL CONTEXT: LOB ABBREVIATIONS & TERMINOLOGY
ABFL: BL=Business Loan, PL=Personal Loan, STUL=Small Ticket Unsecured Loan, STSL=Small Ticket Secured Loan, LAP=Loan Against Property, DIGI PL=Digital Personal Loan, MCA=Merchant Cash Advance, SCF=Supply Chain Finance, PDM=Pre Delinquency Model, EWS=Early Warning Signal, SME=Small Medium Enterprise, MM=Mid Market
ABSLAMC: RFM=Recency Frequency Monetary, STV=Short Term Value
ABSLI: EWI=Early Warning Indicator, RO=Relationship Officer, DM=Direct Marketing
ABHFL: FOS=Feet on Street, HL=Home Loan, LAP=Loan Against Property, NTC=New to Credit

Treat abbreviations and full forms as equivalent. Map user queries to correct terminology before answering. Do not assume meanings beyond what is defined above.

TERMINOLOGY EQUIVALENCE:
- features = attributes = variables = columns = fields
- algorithms = models = techniques = approaches = predictive methods

CRITICAL RULES (NON-NEGOTIABLE):

1. SOURCE CONTROL — STRICT
- Use ONLY retrieved context. Never use external knowledge.
- If exact answer not in context, respond:
  "The requested information is not available in the retrieved documents. Could you please rephrase the question or initiate a fresh chat?"

2. NO BLENDING ACROSS MODELS — CRITICAL
- If retrieved context mentions MULTIPLE models, answer ONLY about the model explicitly asked in the question.
- Do NOT combine facts from different models even if they seem related.
- Example: If asked about "EWI algorithm" and context has EWI + PDM details, answer ONLY EWI's algorithm.
- If the specific model's information is not in context, say so explicitly. Do NOT substitute with another model's data.

3. CITATION DISCIPLINE
- Cite ONLY the document(s) that directly contain the answer.
- Do NOT cite documents that were retrieved but not used.

4. CONTEXT RESOLUTION
- Resolve pronouns (it, this, the model, they, them) using conversation history.
- Internally rewrite ambiguous queries — do not show this to user.
- If ambiguity persists, ask a clarification question.

5. CONFLICT HANDLING
- If documents contain conflicting values, present both and state the discrepancy. Do not decide.

6. MLOPS DEPTH
- For MLOps/framework/lifecycle queries, provide comprehensive detailed answers covering end-to-end flow, key components, dependencies, and supporting processes.
- Synthesize across multiple documents if relevant.
- Never give brief summaries for these queries.

7. TABLE FORMATTING
- For tabular/comparative/multi-column data: use valid HTML tables (<table><thead><tbody><tr><th><td>).
- Never use Markdown pipe tables.
- For missing values write "Not specified in retrieved documents".

8. CONVERSATIONAL TONE — NO FOLLOW-UP QUESTIONS
- Professional analytical tone.
- Answer the question completely in 2-4 sentences for simple queries; structured detail for complex ones.
- Do NOT ask follow-up questions like "Would you like more details?", "Want me to explain further?", "Need more information?", "Shall I elaborate?"
- Do NOT offer to provide additional information.
- Simply provide a complete, thorough answer to what was asked.
- Let the user decide what to ask next.
"""


def build_llm():
    logger.info(f"Using Azure OpenAI LLM: {settings.azure_openai_deployment_name}")
    return AzureChatOpenAI(
        azure_deployment=settings.azure_openai_deployment_name,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        temperature=0,
    )


def build_chain(retriever):
    llm = build_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Context from company documents:\n\n{context}\n\n---\n\n{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    logger.info("RAG chain built successfully")
    return chain


# ──────────────────────────────────────────────────────────────────────────
# Universal Query Rewriter — used by both RAG and Inventory
# ──────────────────────────────────────────────────────────────────────────
def rewrite_query_with_context(question: str, history: list,
                                inventory_filters: dict = None) -> dict:
    """
    Rewrites the question to be self-contained, resolving pronouns
    using conversation history AND inventory filters.

    Returns:
        {
            "rewritten":      "<self-contained question>",
            "is_followup":    bool,
            "is_topic_shift": bool,
        }
    """
    from openai import AzureOpenAI

    if not history and not inventory_filters:
        return {
            "rewritten":      question,
            "is_followup":    False,
            "is_topic_shift": False,
        }

    # Build context block
    context_parts = []
    if history:
        recent = history[-2:]
        hist_lines = []
        for h in recent:
            if isinstance(h, tuple):
                hist_lines.append(f"Q: {h[0]}\nA: {h[1][:200]}")
            elif isinstance(h, dict):
                hist_lines.append(f"Q: {h.get('question', '')}")
        if hist_lines:
            context_parts.append("HISTORY:\n" + "\n".join(hist_lines))

    if inventory_filters:
        flt = ", ".join(f"{k}={v}" for k, v in inventory_filters.items() if v)
        if flt:
            context_parts.append(f"ACTIVE INVENTORY FILTERS: {flt}")

    context_block = "\n\n".join(context_parts) if context_parts else "None"

    rewrite_prompt = f"""You are a query understanding assistant. Rewrite the user's question to be completely self-contained.

{context_block}

NEW QUESTION: "{question}"

Rules:
- If question references "it", "this", "the model", "they", "them", "those", "these" — replace with actual subject from history
- If question references "of them", "of these" — carry forward the previous filters
- If question is already self-contained, return it as-is
- Detect if it's a topic shift (completely new subject)
- Keep rewritten question concise — one sentence

Return ONLY valid JSON:
{{
  "rewritten": "<self-contained question>",
  "is_followup": <true|false>,
  "is_topic_shift": <true|false>
}}

EXAMPLES:

History: Q: "What is EWI model?"
Question: "How does it work?"
Output: {{"rewritten": "How does the EWI model work?", "is_followup": true, "is_topic_shift": false}}

History: Q: "List HR models in ABSLI"
Question: "How many of them are owned by Vaibhav?"
Output: {{"rewritten": "How many HR models in ABSLI are owned by Vaibhav?", "is_followup": true, "is_topic_shift": false}}

History: Q: "What is the algorithm in EWI?"
Question: "Now tell me about RO Hiring model"
Output: {{"rewritten": "Tell me about RO Hiring model", "is_followup": false, "is_topic_shift": true}}
"""

    try:
        client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
        response = client.chat.completions.create(
            model=settings.azure_openai_deployment_name,
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        import json
        result = json.loads(response.choices[0].message.content.strip())
        logger.info(f"Rewritten: '{question}' -> '{result.get('rewritten')}'")
        return result
    except Exception as e:
        logger.warning(f"Query rewrite failed: {e} — using original")
        return {
            "rewritten":      question,
            "is_followup":    False,
            "is_topic_shift": False,
        }


# ──────────────────────────────────────────────────────────────────────────
# Cross-encoder reranking
# ──────────────────────────────────────────────────────────────────────────
def rerank_docs(query: str, docs: list, top_k: int = 5) -> list:
    """
    Rerank retrieved docs using cross-encoder for better relevance.
    Falls back to original order if reranker unavailable.
    """
    reranker = _get_reranker()
    if reranker is None or len(docs) <= top_k:
        return docs[:top_k]

    try:
        pairs = [(query, doc.page_content[:1000]) for doc in docs]
        scores = reranker.predict(pairs)
        scored = list(zip(docs, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        reranked = [doc for doc, _ in scored[:top_k]]
        logger.info(f"Reranked {len(docs)} docs -> top {top_k}")
        return reranked
    except Exception as e:
        logger.warning(f"Rerank failed: {e}")
        return docs[:top_k]


# ──────────────────────────────────────────────────────────────────────────
# Main RAG runner
# ──────────────────────────────────────────────────────────────────────────
def run_rag(chain, retriever, question: str, history: list,
            rewritten_query: str = None) -> tuple[str, list, dict, dict]:
    """
    Run RAG with semantic rewriting + reranking.

    Args:
        rewritten_query: Pre-rewritten standalone query from main.py.
                         If None, falls back to using question as-is.

    Returns:
        (answer, docs, usage_dict, offer_dict_or_None)
    """
    from langchain_core.messages import HumanMessage, AIMessage

    # Convert history tuples to LangChain messages — limit to last 2 turns
    lc_history = []
    for human, ai in history[-2:]:
        lc_history.append(HumanMessage(content=human))
        lc_history.append(AIMessage(content=ai))

    # Use rewritten query for retrieval if provided, otherwise use original
    retriever_query = rewritten_query if rewritten_query else question
    logger.info(f"Retriever query: {retriever_query}")

    # Retrieve more docs than needed, then rerank
    initial_k = max(settings.retrieval_k * 2, 10)
    try:
        # Try to retrieve more for reranking
        retriever.search_kwargs["k"] = initial_k
    except Exception:
        pass

    docs = retriever.invoke(retriever_query)

    # Reset k
    try:
        retriever.search_kwargs["k"] = settings.retrieval_k
    except Exception:
        pass

    # Rerank with cross-encoder
    docs = rerank_docs(retriever_query, docs, top_k=settings.retrieval_k)

    context = "\n\n".join(doc.page_content for doc in docs)

    # Run chain with original question for natural answer flow
    # Chain invocation happens here — LangChain handles token tracking internally
    response_obj = chain.invoke({
        "question": question,
        "context":  context,
        "history":  lc_history,
    })

    # Extract token usage from LangChain response metadata if available
    usage = {
        "prompt_tokens":     0,
        "completion_tokens": 0,
        "total_tokens":      0,
    }
    
    # LangChain's response object may have usage_metadata
    if hasattr(response_obj, 'response_metadata'):
        token_usage = response_obj.response_metadata.get('token_usage', {})
        usage = {
            "prompt_tokens":     token_usage.get('prompt_tokens', 0),
            "completion_tokens": token_usage.get('completion_tokens', 0),
            "total_tokens":      token_usage.get('total_tokens', 0),
        }
    
    # If response_obj is a string (from StrOutputParser), we don't have metadata
    # In that case, estimate tokens (rough approximation: 1 token ≈ 4 chars)
    if isinstance(response_obj, str):
        response_text = response_obj
        # Rough estimation
        est_completion = len(response_text) // 4
        est_prompt = (len(question) + len(context)) // 4
        usage = {
            "prompt_tokens":     est_prompt,
            "completion_tokens": est_completion,
            "total_tokens":      est_prompt + est_completion,
        }
    else:
        response_text = response_obj

    # Detect if answer contains an offer
    offer = None
    answer_lower = str(response_text).lower()
    if any(phrase in answer_lower for phrase in [
        "would you like", "want me to", "shall i", "do you want",
        "interested in", "need more", "like to know more",
        "would that help", "want to explore", "let me know if"
    ]):
        # Extract what was being discussed (from last question or model name in context)
        subject = "this topic"
        if history:
            last_q = history[-1][0] if history[-1] else ""
            # Try to extract model name or topic from last question
            if "model" in last_q.lower():
                subject = last_q
        
        offer = {
            "type": "more_details",
            "subject": subject,
        }

    return str(response_text).strip(), docs, usage, offer
