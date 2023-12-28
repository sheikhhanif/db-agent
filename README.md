# Langchain SQL Agent Project

## Overview
This project utilizes Langchain integrated with OpenAI's GPT-4 to create a sophisticated SQL agent. It features custom tools for efficient query handling, leveraging advanced NLP techniques and vectorized search for robust data retrieval.

## Key Features
- **Custom SQL Agent**: Built on Langchain and OpenAI's GPT-4 for advanced query processing.
- **HuggingFace Embeddings**: Uses HuggingFace for nuanced text representation.
- **FAISS Vector Stores**: Employs FAISS for efficient similarity searches in high-dimensional data.
- **Specialized Retrieval Tools**: Custom tools for tailored query and data retrieval.

## Custom Tools
1. **SQL Query Retriever**: Fetches relevant SQL queries based on user inputs, aiding in dynamic response generation.
2. **Name Search Retriever**: Identifies accurate data representations, crucial for transactions and merchant data.

## Requirements
- Python 3.x
- Langchain
- OpenAI GPT-4 (API key required)
- HuggingFace's Transformers
- FAISS
- Pandas

To install dependencies:
```bash
pip install langchain openai huggingface_hub faiss-cpu pandas
