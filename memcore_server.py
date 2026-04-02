import json
import logging
from mcp.server.fastmcp import FastMCP
from memcore import MemCoreEngine # Our compiled Rust library
import threading
import asyncio
from consolidation_worker import consolidation_loop

import os
log_file = os.path.expanduser("~/.hermes/memcore.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=log_file,
    filemode='a'
)
logger = logging.getLogger("memcore-mcp")

try:
    from sentence_transformers import SentenceTransformer
    # MiniLM generates perfectly sized 384-dim arrays natively
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Local Embedding Model (MiniLM-L6-v2) loaded successfully.")
except ImportError:
    embedder = None
    logger.warning("sentence-transformers missing. Run `pip install sentence-transformers` to generate LanceDB vectors.")

mcp = FastMCP("MemCore-Harness")
engine = MemCoreEngine()

@mcp.tool()
def commit_memory(
    text: str, 
    agent_id: str, 
    epistemic_uncertainty: float = 0.1,
    temperature: float = 7.0
) -> str:
    """
    Mandatory harness operation for storing agent memory.
    The harness mathematically enforces graph acyclicity, volume fractions, and non-oscillation.
    """
    # Define S = {T, O, C}
    skill_tuple = {
        "t": "PassThrough",
        "o": ["InsertNode"],
        "c": [
            {"NoOscillation": {"window_size": 3}},
            {"EvalScript": "text_len > 10"} # Force agent to write substantive memory
        ]
    }
    
    # Physics Observation
    dnc_obs = {
        "grayness_index": epistemic_uncertainty,
        "stagnation_counter": 0,
        "budget_consumption": 0.5, # 50% memory used
        "relative_change": 0.1,
        "checkerboard_measure": 0.0
    }

    # Generate local embeddings via MiniLM locally if requested
    embedding_json = None
    if embedder is not None:
        raw_emb = embedder.encode(text).tolist()
        embedding_json = json.dumps(raw_emb)

    try:
        # Executes in Rust (< 1 millisecond)
        result = engine.store(
            agent_id=agent_id,
            text=text,
            skill_tuple_json=json.dumps(skill_tuple),
            dnc_obs_json=json.dumps(dnc_obs),
            temperature=temperature,
            embedding_json=embedding_json
        )
        return f"[SUCCESS] {result}"
        
    except ValueError as e:
        # ⚠️ CLOSING THE VALIDITY GAP:
        # If the Rust physics engine rejects the state, we return the error
        # directly back to the LLM agent via MCP. The LLM sees the text error
        # and inherently self-corrects its reasoning without crashing the framework.
        error_msg = str(e)
        logger.warning(f"Memory rejected: {error_msg}")
        return f"[HARNESS REJECTION] {error_msg}\nAction Required: Re-evaluate your reasoning."

@mcp.tool()
def query_semantic_memory(query_text: str, limit: int = 5) -> str:
    """
    Search your memories via Approximate Nearest Neighbor vector distance.
    The Rust physics engine will return exact memory strings matching your query.
    """
    if embedder is None:
        return "[ERROR] sentence-transformers not loaded. Cannot generate embeddings."
    
    try:
        raw_emb = embedder.encode(query_text).tolist()
        result = engine.query_semantic_memory(json.dumps(raw_emb), limit)
        return f"[SEARCH RESULTS]\n{result}"
    except Exception as e:
        return f"[SEARCH FAILED] {str(e)}"

@mcp.tool()
def traverse_knowledge_graph(cypher_query: str) -> str:
    """
    Direct read-access to the Kùzu property graph for structural correlation finding.
    Write a Cypher query (e.g., 'MATCH (m:Memory) RETURN m LIMIT 5').
    """
    try:
        result = engine.traverse_knowledge_graph(cypher_query)
        if not result.strip():
            return "[GRAPH] Query executed successfully, but returned zero rows."
        return f"[GRAPH RESULTS]\n{result}"
    except Exception as e:
        return f"[GRAPH QUERY FAILED] {str(e)}"

@mcp.resource("memory://core_index")
def read_core_index() -> str:
    """Read the Layer 1 Cognitive Index of all critical pointers."""
    try:
        # We query the Kùzu graph for pointers that are either Critical urgency or have a negative Valence score
        query = "MATCH (m:Memory) WHERE m.urgency = 'Critical' OR m.valence <= -0.5 RETURN m.pointer"
        result = engine.traverse_knowledge_graph(query)
        if not result.strip():
            return "Core Index is currently empty."
        return result
    except Exception as e:
        return f"[ERROR reading Core Index] {str(e)}"

def start_background_loop(engine_ref):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(consolidation_loop(engine_ref))

if __name__ == "__main__":
    logger.info("Starting AutoDream Background Daemon...")
    worker = threading.Thread(target=start_background_loop, args=(engine,), daemon=True)
    worker.start()
    
    logger.info("MemCore v0.4 MCP Server active. Awaiting connections...")
    mcp.run()
