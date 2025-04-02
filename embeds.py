import PyPDF2
import faiss
import numpy as np
from google import genai
from google.genai import types
from api_keys import API_KEY
import os
import json

from sentence_transformers import SentenceTransformer
import numpy as np

client = genai.Client(api_key=API_KEY)

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    lines = []
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            lines.extend(text.splitlines())
    return lines

# def get_embeddings(text):

#     result = client.models.embed_content(
#             model="gemini-embedding-exp-03-07",
#             contents=text,
#             config=types.EmbedContentConfig(
#                 task_type="RETRIEVAL_QUERY",
#                 output_dimensionality=768,
#                 ),
#             )
#     return np.array([embedding.values for embedding in result.embeddings], dtype=np.float32)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embeddings(text):
    embeddings = model.encode(text, convert_to_numpy=True)
    return np.array(embeddings, dtype=np.float32)

def generate_and_store_embeddings(files):
    index_file_path = "embeddings.index"
    json_file_path = "mappings.json"

    if os.path.exists(index_file_path):
        index = faiss.read_index(index_file_path)
        with open(json_file_path, "r") as f:
            line_map = json.load(f)
        print("Loaded pre-existing FAISS index and mappings")
    else:
        index = faiss.IndexFlatL2(384)
        line_map = {}
        print("Initialising new FAISS index and mappings")
        
    total=[]
    db_id = len(line_map)

    for file in files:
        extracted_text = extract_text_from_pdf(file)
        for i,line in enumerate(extracted_text):
            text_embeddings = get_embeddings(line)

            total.append(text_embeddings)

            line_map[str(db_id)] = i,file
            db_id += 1

    all_embeddings = np.vstack(total)
    
    index.add(all_embeddings)
    faiss.write_index(index, index_file_path)

    with open(json_file_path, "w") as f:
        json.dump(line_map, f, indent=4)

if __name__ == "__main__":
    files =["sample.pdf"]
    generate_and_store_embeddings(files)