import streamlit as st
import os
import xml.etree.ElementTree as ET
import feedparser

from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import get_openai_callback
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import LanceDB

#Configuramos el Streamlit
st.set_page_config('PDFChat', page_icon=":page_facing_up")
st.header("¿Qué vamos aprender hoy?")

os.environ['OPENAI_API_KEY'] = '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'

obj_pdf = st.file_uploader("Carga tu Documento", type=["html", "rss", "pdf", "xml"], accept_multiple_files=True, on_change=st.cache_resource.clear)

@st.cache_resource
def crear_embeddings(pdf):
    if obj_pdf:
        text = ""
        for uploaded_file in obj_pdf:
            if uploaded_file.type == "text/html":
                text += uploaded_file.read().decode("utf-8")
            elif uploaded_file.type == "application/rss+xml":
                rss_content = feedparser.parse(uploaded_file)
                for entry in rss_content.entries:
                    text += entry.title + " " + entry.description + " "
            elif uploaded_file.type == "application/pdf":
                with uploaded_file as pdf_file:
                    pdf_reader = PdfReader(pdf_file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()

    #Divimos en chunks
    dividir_text = RecursiveCharacterTextSplitter(
        chunk_size= 800,
        chunk_overlap=100,
        length_function=len
    )
    chunks = dividir_text.split_text(text)

    #se crea los embeddings y se almacenan en una base
    crear_embeddings = OpenAIEmbeddings()
    bd_embedding = LanceDB.from_texts(chunks, crear_embeddings)

    return bd_embedding, text

if obj_pdf:
    bd_embeddings, texto_documento = crear_embeddings(obj_pdf)
    u_pregunta = st.text_input("¿Cuales es la pregunta?")

    if st.button("Responde"):
        #Realiza la busqueda de similitud y procesa la pregunta
        docs = bd_embeddings.similarity_search(u_pregunta, 3)
        llm = OpenAI(model_name='gpt-3.5-turbo-instruct')
        chain = load_qa_chain(llm, chain_type="stuff")
        respuesta = chain.run(input_documents=docs, question=u_pregunta)
        st.write("Respuesta Generada: ", respuesta)

        with get_openai_callback() as cost:
            response = chain.invoke(input={"question": u_pregunta, "input_documents": docs})
            print(cost)

            st.write(response["output_text"])
            st.write("Costo de la Operacion:", cost)