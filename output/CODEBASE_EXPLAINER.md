# Codebase Explanation

## Overview

- **Total Components**: 11
- **Total Dependencies**: 11


## Component Details

### File: `/home/deeog/Desktop/claude-dementia/server.py`

#### class `Config`

- **Location**: Line 31

**Core Purpose**: The `Config` class is designed to encapsulate configuration settings for the server application, specifically related to external services and database paths. It initializes these settings by reading environment variables or defaulting to predefined values if the environment variables are not set. **Key Inputs & Outputs**: - **Inputs**: Environment variables (`OLLAMA_BASE_URL`, `EMBEDDING_MODEL`, `CLAUDE_MEMORY_DB`). - **Outputs**: Instance attributes (`ollama_base_url`, `embedding_model`, `db_path`) initialized with values from the environment or defaults. **Dependencies**: - The class relies on the `os` module for accessing environment variables. **Usage Patterns**: - Typically instantiated at the start of the application to load configuration settings. These settings are then used throughout the server code to interact with external services and manage database operations.

---

#### function `__init__`

- **Location**: Line 32

**Core Purpose**: Initializes configuration settings for the memory server, including Ollama base URL, embedding model, and database path. **Key Inputs & Outputs**: No inputs; sets instance variables based on environment variables or default values. **Dependencies**: Relies on `os` module to access environment variables. **Usage Patterns**: Called when an instance of the server class is created.

---

#### function `get_db_connection`

- **Location**: Line 43

**Core Purpose**: Establishes a connection to the SQLite database configured for the memory server. **Key Inputs & Outputs**: No inputs; returns a SQLite database connection object. **Dependencies**: Relies on `sqlite3` module and `config` for the database path. **Usage Patterns**: Called by various functions that require database interaction.

---

#### function `initialize_database`

- **Location**: Line 49

**Core Purpose**: Initializes the necessary tables in the SQLite database if they do not already exist. **Key Inputs & Outputs**: No inputs; creates and commits database schema changes. **Dependencies**: Relies on `get_db_connection` to establish a database connection. **Usage Patterns**: Called during server initialization to ensure the database is properly set up.

---

#### function `get_embedding`

- **Location**: Line 94

**Core Purpose**: Generates an embedding vector for a given text using Ollama's API. **Key Inputs & Outputs**: Takes a string `text`; returns a list of floats representing the embedding or `None` if generation fails. **Dependencies**: Relies on `httpx` for HTTP requests, `config` for Ollama settings, and handles exceptions to provide error messages. **Usage Patterns**: Called by functions that require text embeddings, such as storing and searching memories.

---

#### function `get_status`

- **Location**: Line 120

**Core Purpose**: Provides the current status of the memory server, including database path and embedding model. **Key Inputs & Outputs**: No inputs; returns a string with server status information. **Dependencies**: Relies on `config` for configuration settings. **Usage Patterns**: Called to retrieve server status, useful for debugging or logging.

---

#### function `store_memory`

- **Location**: Line 125

**Core Purpose**: Stores a memory entry in the database, including generating an embedding for the content if available. **Key Inputs & Outputs**: Takes `content`, `label`, `is_persistent`, and `project_path`; returns a success or error message. **Dependencies**: Relies on `get_db_connection` for database operations, `get_embedding` to generate embeddings, and handles exceptions for robustness. **Usage Patterns**: Called when storing new memories in the server.

---

#### function `retrieve_memory`

- **Location**: Line 178

**Core Purpose**: Retrieves a specific memory entry from the database based on its label. **Key Inputs & Outputs**: Takes `label` and optional `project_path`; returns the content of the memory or an error message if not found. **Dependencies**: Relies on `get_db_connection` for database operations. **Usage Patterns**: Called when retrieving memories by their labels.

---

#### function `search_memories`

- **Location**: Line 195

**Core Purpose**: Searches for memories in the database using vector similarity if embeddings are available, otherwise falls back to text search. **Key Inputs & Outputs**: Takes `query`, `limit`, and optional `project_path`; returns a list of matching memory entries or an error message if none found. **Dependencies**: Relies on `get_db_connection` for database operations, `get_embedding` to generate embeddings, `cosine_similarity` for scoring, and handles exceptions for robustness. **Usage Patterns**: Called when searching for memories based on query text.

---

#### function `cosine_similarity`

- **Location**: Line 219

**Core Purpose**: Computes the cosine similarity between two vectors. **Key Inputs & Outputs**: Takes two vectors `v1` and `v2`; returns a float representing their cosine similarity. **Dependencies**: Relies on `numpy` for vector operations. **Usage Patterns**: Called during vector search to score memory content based on query relevance.

---

### File: `/home/deeog/Desktop/claude-dementia/verify_local.py`

#### function `verify`

- **Location**: Line 7

**Core Purpose**: The `verify` function is designed to check the integrity and functionality of a local Memory Control Protocol (MCP) server by verifying the database, Ollama embeddings, memory storage, and vector search capabilities. **Key Inputs & Outputs**: No direct inputs; outputs are print statements indicating the status of each verification step. **Dependencies**: Relies on `config` for configuration settings, `sqlite3` for database operations, `get_embedding`, `store_memory`, `retrieve_memory`, and `search_memories` functions from `server.py`. **Usage Patterns**: Typically called during server startup or manually to ensure the MCP server is functioning correctly.

---
