from google import genai
from google.genai import types
from api_keys import API_KEY
from embeds import get_embeddings
from embeds import extract_text_from_pdf
import streamlit as st
import time
import PyPDF2
import faiss
import json

client = genai.Client(api_key=API_KEY)

def semantic_search(query, k=3):
    query_embeddings = get_embeddings(query)
    query_embeddings = query_embeddings.reshape(1, -1)
    print(query_embeddings.shape)
    index = faiss.read_index("embeddings.index")
    print("Loading embeddings")

    with open("mappings.json", "r") as f:
        line_map = json.load(f)

    D, I = index.search(query_embeddings, k)

    results = []

    for idx in I[0]:
        if str(idx) in line_map:  
            i,file_path = line_map[str(idx)]
            extracted_text = extract_text_from_pdf(file_path)
            
            excerpt = extracted_text[i-1] + extracted_text[i] + extracted_text[i+1]
            results.append({"file": file_path, "excerpt": excerpt, "distance": D[0][list(I[0]).index(idx)]})

    return results

def generate_answers(excerpts, query):
    instructions = f"You are given a set of excerpts from relevant research papers : {excerpts}, when prompted with a question related to these excerpts, provide an answer using mostly the set given, give the output in proper markdown format"
    response_stream = client.models.generate_content_stream(
        model="gemini-2.0-flash", 
        contents=query,
        config=types.GenerateContentConfig(
            system_instruction=instructions
            ),
    )
    
    response_text = ""
    response_placeholder = st.empty()

    for chunk in response_stream:
        for char in chunk.text:
            response_text += char
            response_placeholder.markdown(response_text)
            time.sleep(0.02)

def ask_query(query):
    semantic_result = semantic_search(query)
    generate_answers([x["excerpt"] for x in semantic_result],query)
    print([x["excerpt"] for x in semantic_result])


st.title("Research Paper Q&A")
query = st.text_input("Enter your query:")
if st.button("Ask"):
    ask_query(query)
