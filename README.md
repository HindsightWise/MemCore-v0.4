# MemCore v0.4
**Autonomous Zero-Copy Memory Engine for Enterprise AI Agents**

MemCore is a hybrid Rust/Python memory persistence architecture designed explicitly for autonomous reasoning agents (like Hermes, Claude Desktop, and Letta). It bypasses the massive overhead of traditional Python-based memory systems, allowing heavy memory persistence natively on Apple Silicon without triggering memory saturation or kernel panics (strictly under a ≤ 9GB peak footprint).

## 🚀 Core Architecture

The architecture utilizes a "Sandwich" model: all heavy numerical computation and constraint checks execute asynchronously in sub-millisecond Rust `cdylib` extensions, while Python acts purely as the routing harness (via `FastMCP`).

1. **Semantic Legislator (`Rhai`)**: Every memory attempt is logically audited by an embedded Rust scripting engine (`rhai`) before being considered. 
2. **Direct Numeric Control (DNC)**: A mathematical logic gate that manages Epistemic Uncertainty and Structural Entropy to determine if a memory is a hallucination or an actual fact.
3. **LanceDB Vectors**: Passes 384-dimensional `f32` sentence embeddings straight into native Arrow `RecordBatches` flushed to `.lance` datasets for hyper-fast semantic search.
4. **KùzuDB Property Graphs**: Creates zero-copy logical relationships (Nodes & Edges) inside a localized graph database representing the agent's current state.

## 🛠️ Installation & Compilation

Since the core relies on extreme memory limits, everything must be compiled to your native architecture via `maturin`. 

### Prerequisites
- Rust & Cargo (`rustup default stable`)
- Python 3.10+
- CMake (minimum 3.5 required for C++ graph driver compilation)

### Building the Core
```bash
# 1. Activate your virtual environment
source venv/bin/activate

# 2. Install the local embedding generator 
pip install sentence-transformers

# 3. Compile the Rust Extension in Release Mode
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
```

## 🧠 Connecting Your Agent

MemCore exposes itself as a standard **Model Context Protocol (MCP)** server over `stdio`. 

Append the following command to your agent framework (example using Hermes CLI):
```bash
hermes mcp add memcore --command /path/to/venv/bin/python --args /path/to/memcore/memcore_server.py
```

### Available Tools
- `commit_memory`: The agent uses this tool to store semantic strings and embeddings. 

## 📂 Data Storage
All databases automatically construct themselves independently of your agent's main branch inside your system user's root folder:
- **Graph:** `~/.hermes/memcore_data/kuzu_graph/`
- **Vectors:** `~/.hermes/memcore_data/lancedb/`

---
*Built as the Memory Substrate for Project Chimera / The Company Architecture.*
