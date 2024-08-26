from src.helper import load_pdf, text_split, download_hugging_face_embeddings,download_openai_embeddings
from langchain.vectorstores import FAISS
import numpy as np


extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
#embeddings = download_hugging_face_embeddings()
embeddings = download_openai_embeddings()

#vectorstore
vector_store = FAISS.from_documents(text_chunks,embeddings)

# Save the FAISS index to disk
vector_store.save_local("faiss_index_openai")