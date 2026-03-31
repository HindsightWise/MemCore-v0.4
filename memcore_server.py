import json
import logging
from mcp.server.fastmcp import FastMCP
from memcore import MemCoreEngine # Our compiled Rust library

logging.basicConfig(level=logging.INFO)
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

if __name__ == "__main__":
    logger.info("MemCore v0.4 MCP Server active. Awaiting connections...")
    mcp.run()
