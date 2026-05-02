"""
Helper functions for the Vista Victoria by-laws RAG pipeline.

Mirrors the structure of the DeepLearning.ai Advanced RAG course utils.py
but uses:
  - llama_index.core  (new 0.11+ API, Settings replaces ServiceContext)
  - Claude via llama-index-llms-anthropic as the RAG LLM
  - Ragas for evaluation, also driven by Claude (no extra API key needed)
"""

import os
import nest_asyncio
from dotenv import load_dotenv, find_dotenv

nest_asyncio.apply()


# ---------------------------------------------------------------------------
# API key helper
# ---------------------------------------------------------------------------

def get_anthropic_api_key() -> str:
    load_dotenv(find_dotenv())
    return os.getenv("ANTHROPIC_API_KEY", "")


# ---------------------------------------------------------------------------
# Ragas evaluation
#
# Ragas triad (maps 1-to-1 with the TruLens triad from the course):
#   faithfulness        ≈ Groundedness    — every claim backed by retrieved text
#   answer_relevancy    ≈ Answer Relevance — answer actually addresses the question
#   context_precision   ≈ Context Relevance — retrieved chunks are on-topic
#
# All three work without ground-truth answers, so no labelling required.
# ---------------------------------------------------------------------------

def evaluate_pipeline(query_engine, questions: list[str]) -> "pandas.DataFrame":
    """
    Run `questions` through `query_engine`, then score each response using
    Claude Haiku as judge.  Returns a pandas DataFrame with columns:
        question, answer, faithfulness, answer_relevancy

    faithfulness    — is every claim in the answer supported by the retrieved chunks?
    answer_relevancy — does the answer actually address the question?

    Both scored 0.0–1.0.  Two API calls per question (one per metric).
    """
    import anthropic
    import pandas as pd

    client = anthropic.Anthropic()
    rows = []

    for q in questions:
        response = query_engine.query(q)
        answer = str(response)
        context_text = "\n---\n".join(node.text for node in response.source_nodes)

        faith_resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=5,
            messages=[{
                "role": "user",
                "content": (
                    "Rate how faithfully this answer is grounded in the provided context.\n"
                    "Score 0.0 (answer adds claims not in context) to 1.0 (every claim is supported).\n"
                    f"Context:\n{context_text}\n\nAnswer:\n{answer}\n\n"
                    "Reply with a single decimal number only."
                ),
            }],
        )
        try:
            faithfulness = float(faith_resp.content[0].text.strip())
        except ValueError:
            faithfulness = float("nan")

        rel_resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=5,
            messages=[{
                "role": "user",
                "content": (
                    "Rate how well this answer addresses the question.\n"
                    "Score 0.0 (off-topic) to 1.0 (directly answers the question).\n"
                    f"Question: {q}\nAnswer:\n{answer}\n\n"
                    "Reply with a single decimal number only."
                ),
            }],
        )
        try:
            answer_relevancy = float(rel_resp.content[0].text.strip())
        except ValueError:
            answer_relevancy = float("nan")

        rows.append({
            "question": q,
            "answer": answer,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sentence-window index + query engine
# ---------------------------------------------------------------------------

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core import load_index_from_storage
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)


def build_sentence_window_index(
    document,
    llm,
    embed_model,
    save_dir="sentence_index",
):
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = node_parser

    if not os.path.exists(save_dir):
        index = VectorStoreIndex.from_documents([document])
        index.storage_context.persist(persist_dir=save_dir)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir)
        )

    return index


def get_sentence_window_query_engine(
    sentence_index,
    similarity_top_k=3,
    rerank_top_n=2,
    use_reranker=False,
):
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    postprocessors = [postproc]
    if use_reranker:
        rerank = SentenceTransformerRerank(
            top_n=rerank_top_n, model="BAAI/bge-reranker-base"
        )
        postprocessors.append(rerank)
    return sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k,
        node_postprocessors=postprocessors,
    )


# ---------------------------------------------------------------------------
# Auto-merging index + query engine
# ---------------------------------------------------------------------------

from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine


def build_automerging_index(
    documents,
    llm,
    embed_model,
    save_dir="merging_index",
    chunk_sizes=None,
):
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)

    Settings.llm = llm
    Settings.embed_model = embed_model

    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)
        index.storage_context.persist(persist_dir=save_dir)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir)
        )

    return index


def get_automerging_query_engine(
    automerging_index,
    similarity_top_k=6,
    rerank_top_n=2,
    use_reranker=False,
):
    base_retriever = automerging_index.as_retriever(
        similarity_top_k=similarity_top_k
    )
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    postprocessors = []
    if use_reranker:
        rerank = SentenceTransformerRerank(
            top_n=rerank_top_n, model="BAAI/bge-reranker-base"
        )
        postprocessors.append(rerank)
    return RetrieverQueryEngine.from_args(retriever, node_postprocessors=postprocessors)
