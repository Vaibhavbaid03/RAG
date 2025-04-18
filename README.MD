# Project Description (RAG-based Help Assistant)
This project implements a Retrieval-Augmented Generation (RAG) chatbot using:

- Qdrant (Cloud) for storing vector embeddings
- Hugging Face (or Ollama) for generating text embeddings
- Ollama + Mistral for answering questions using local LLM
- LangChainJS + Fastify for chaining and API handling

The bot indexes 160 Refrens Help Center articles from https://help.refrens.com/sitemap.xml, embeds their content into Qdrant, and uses it to answer natural language questions via an HTTP POST /rag endpoint.

 - On first run, it loads and embeds all articles in batches
 - On next runs, it skips re-embedding and serves answers instantly
---

##  Project Structure

```
├── server.ts               # Fastify server exposing the /rag POST endpoint
├── ragHandler.ts           # Core RAG logic: loads URLs, chunks content, embeds vectors, retrieves answers
├── Refrens_Help_URLs.csv   # List of Refrens help URLs to be scraped & embedded
├── .env                    # Environment variables (Hugging Face & Qdrant API keys)
├── tsconfig.json           # TypeScript configuration
```


##  Environment Setup


## 1. Clone and Install
    ```bash
    git clone https://github.com/your-org/refrens-rag-chatbot.git
    cd refrens-rag-chatbot
    npm install

. 
## 2. Setup.env
    Create a .env file:
    HUGGINGFACE_API_KEY=your_hf_key
    QDRANT_API_KEY=your_qdrant_key

Get your API keys from:
HuggingFace: https://huggingface.co/settings/tokens
Qdrant Cloud: https://cloud.qdrant.io


## 3.Install & Run Ollama

    curl -fsSL https://ollama.com/install.sh | sh

Then pull & run Mistral locally:

    ollama pull mistral
    ollama run mistral

##4. Running the App

Step 1: Compile (optional)

    npx tsc

Step 2: Start the Fastify Server

    npx tsx server.ts


## How to query:

    curl -X POST http://localhost:3000/rag \
    -H "Content-Type: application/json" \
    -d '{ "question": "How to delete business?" }'


## Dependencies:

- npm install langchain @langchain/community @langchain/core
- npm install fastify dotenv node-fetch
- npm install @qdrant/js-client-rest
- npm install csv-parser
- npm install tsx typescript --save-dev



## Features:

 - Batch loads and embeds 100+ Refrens articles from CSV 
 - HuggingFace Embeddings (MiniLM)
 - Qdrant Cloud with HNSW + on-disk payloads 
 - Mistral via Ollama for answer generation 
 - API endpoint: POST /rag 
 - Accurate source-citation in responses
