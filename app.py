from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
# from langchain.llms import CTransformers
from langchain.chains import RetrievalQA,LLMChain
from src.prompt import *
import os
import numpy as np
from src.helper import download_openai_embeddings
from langchain_community.chat_models import ChatOpenAI



os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"


app = Flask(__name__)

embeddings = download_openai_embeddings()



# Load the FAISS index from disk
vector_store = FAISS.load_local("faiss_index_openai", embeddings,allow_dangerous_deserialization=True)


# PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
PROMPT = PromptTemplate.from_template(prompt_template)
chain_type_kwargs={"prompt": PROMPT}

llm = ChatOpenAI(model_name = 'gpt-3.5-turbo',temperature=0)
# llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
#                   model_type="llama",
#                   config={'max_new_tokens':512,
#                           'temperature':0.8})

llm_chain = LLMChain(llm=llm,prompt=PROMPT)

# Set up the retriever from the FAISS vector store
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# Set up the RetrievalQA chain
# qa=RetrievalQA.from_chain_type(
#     llm=llm, 
#     chain_type="stuff", 
#     retriever=retriever,
#     chain_type_kwargs = chain_type_kwargs)


# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)



@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    question = msg
    print(input)
    result=qa_chain({"query":question})
    print("Response : ", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)