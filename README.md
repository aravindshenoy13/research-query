# Research Query Tool

An RAG-based tool for querying research papers using semantic search and NLP. Upload research papers, enter a question and recieve paper-specific answers from the LLM. Uses FAISS for vector search, Gemini API for response generation, and Streamlit for a simple UI.

## Get Started
```
$ git clone https://github.com/aravindshenoy13/research-query.git
$ cd research-query
$ streamlit run app.py
```

## How It Works

1. **Upload Research Papers**: Users can upload relevant research papers for querying.

2. **Embedding Process**: The tool generates and stores embeddings for efficient semantic search.

3. **Semantic Search**: Retrieves the most relevant excerpts from research papers using FAISS.

4. **Query & Response**: NLP and Gemini API generate answers based on retrieved excerpts.

## Technologies Used

- **NLP Libraries**: For text processing and analysis.
- **FAISS**: For storing the vector embeddings.
- **Gemini API**: Provides prompt-engineered suggestions for improvements.
- **Streamlit**: Frontend for deploying and accessing the tool.

