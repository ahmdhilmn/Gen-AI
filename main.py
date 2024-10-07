from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
#from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from textgen import TextGen

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.requests import Request

from pydantic import BaseModel, Field
import os, re

class RAGLlm(BaseModel):
    model_url: str | None = Field(default="http://103.251.2.17:5000")
    context: str | None = Field(default='data/orient-context.pdf', description="PDF file path on local. (eg. orient-context.pdf, webermeyer-context.pdf)")
    prompt: str | None = Field(default=None, description="User input")

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.options("/")
async def options_route():
    return JSONResponse(content="OK")

@app.post("/rag", summary="Production RAG")
async def rag(request: Request, request_data: RAGLlm):
    """
        Naive RAG implementation with no data persistence.
    """
    
    prompt_template = """
    ### [INST] Instruction: Give only greetings if there is no question. Answer the question based on the context information and if the question can't be answered based on the context, say "I don't know". Here is context to help:

    {context}

    ### QUESTION:
    {question} [/INST]
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    textgen_llm = TextGen(model_url=request_data.model_url, mode="instruct", temperature=0.1, repetition_penalty=1.1, max_new_tokens=1000, truncation_length=32768, do_sample=True)

    loader = PyPDFLoader(request_data.context)
    pages = loader.load_and_split()
    
    db = FAISS.from_documents(pages, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
    # jinaai/jina-embeddings-v2-base-en, sentence-transformers/all-mpnet-base-v2, all-MiniLM-L6-v2, intfloat/e5-large-v2

    retriever_pdf = db.as_retriever(
        search_kwargs={'k': 5},
        # search_type="similarity",
    )

    rag_chain = ( 
    {"context": retriever_pdf, "question": RunnablePassthrough()}
        | prompt
        | textgen_llm
    )
    print(pages)
    response = rag_chain.invoke(f"{request_data.prompt}")
    return response

@app.post("/v2/rag", summary="Testing RAG")
async def rag2(request: Request, request_data: RAGLlm):
    """
        Naive RAG with added functionality of data persistence in local file ./store/
    """

    prompt_template = """
    ### [INST] Instruction: Give only greetings if there is no question. Answer the question based on the context information and if the question can't be answered based on the context, say "I don't know". Here is context to help:

    {context}

    ### QUESTION:
    {question} [/INST]
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )
 
    match = re.search(r'\/([^\/]+)\.pdf$', request_data.context)
    doc_name = match.group(1) if match else None
    doc_dir = f'./store/{doc_name}'
    
    if os.path.isdir(doc_dir):
        print(f"Data file: '{doc_name}' already existed.")
        db = FAISS.load_local(doc_dir, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
        # jinaai/jina-embeddings-v2-base-en, sentence-transformers/all-mpnet-base-v2, all-MiniLM-L6-v2, intfloat/e5-large-v2
    
    else:
        print(f"Data file: '{doc_name}' does not exists.")
        loader = PyPDFLoader(request_data.context)
        pages = loader.load_and_split()
        db = FAISS.from_documents(pages, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
        db.save_local(doc_dir)

    textgen_llm = TextGen(model_url=request_data.model_url, mode="instruct", temperature=0.1, repetition_penalty=1.1, max_new_tokens=1000, truncation_length=32768, do_sample=True)

    retriever_pdf = db.as_retriever(search_kwargs={'k': 5})
    
    rag_chain = ( 
    {"context": retriever_pdf, "question": RunnablePassthrough()}
        | prompt
        | textgen_llm
    )

    response = rag_chain.invoke(f"{request_data.prompt}")
    return response