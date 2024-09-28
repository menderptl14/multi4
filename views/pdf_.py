import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai.llms import _BaseGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_astradb import AstraDBVectorStore
# from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
# import pandas as pd
# from ecommbot.data_converter import dataconveter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # db = Chroma.from_documents(documents,OpenAIEmbeddings())
    # vector_store = Chroma.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    
    # vector_store = FAISS(
    # embedding_function=embeddings,
    # index=index,
    # docstore=InMemoryDocstore(),
    # index_to_docstore_id={},
    # )

#     vector_store = Chroma(
#          collection_name="example_collection",
#          embedding_function=embeddings,
#          persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
# )

    # collection_name="example_collection",
    # embedding_function=embeddings,
    # persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    # model =GoogleGenerativeAI(model="gemini-1.0-pro",
    #                          temperature=0.3)
    
    model = GoogleGenerativeAI(_BaseGoogleGenerativeAI, "gemini-1.0-pro")
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
  
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    # st.write("Reply: ", response["output_text"])


def main():
    # st.set_page_config("Chat PDF")
    # st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    # with st.sidebar:
    #     # st.title("Menu:")
    #     pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    #     if st.button("Submit & Process"):
    #         with st.spinner("Processing..."):
    #             raw_text = get_pdf_text(pdf_docs)
    #             text_chunks = get_text_chunks(raw_text)
    #             get_vector_store(text_chunks)
    #             st.success("Done")

if __name__ == "__main__":
    main()


# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
#     new_db = FAISS.load_local("faiss_index", embeddings)
#     docs = new_db.similarity_search(user_question)

#     chain = get_conversational_chain()

#     response = chain(
#         {"input_documents":docs, "question": user_question}
#         , return_only_outputs=True)

#     print(response)
#     st.write("Reply: ", response["output_text"])


# ASTRA_DB_API_ENDPOINT=os.getenv("ASTRA_DB_API_ENDPOINT")
# ASTRA_DB_APPLICATION_TOKEN=os.getenv("ASTRA_DB_APPLICATION_TOKEN")
# ASTRA_DB_KEYSPACE=os.getenv("ASTRA_DB_KEYSPACE")

# # embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# def get_pdf_text(pdf_docs):
#     docs=""
#     for pdf in pdf_docs:
#         pdf_reader= PdfReader(pdf)
#         for page in pdf_reader.pages:
#             docs+= page.extract_text()
#     return  docs

# embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# def ingestdata(status):
#     vstore = AstraDBVectorStore(
#             embedding=embedding,
#             collection_name="PDF_SUMMARISE",
#             api_endpoint=ASTRA_DB_API_ENDPOINT,
#             token=ASTRA_DB_APPLICATION_TOKEN,
#             namespace=ASTRA_DB_KEYSPACE,
#         )
    
#     storage=status
    
#     if storage==None:
#         # docs=dataconveter()
#         inserted_ids = vstore.add_documents(docs)
#     else:
#         return vstore
#     return vstore, inserted_ids

# if __name__=='__main__':
#     vstore,inserted_ids=ingestdata(None)
#     print(f"\nInserted {len(inserted_ids)} documents.")
#     results = vstore.similarity_search("can you tell me the low budget sound basshead.")
#     for res in results:
#             print(f"* {res.page_content} [{res.metadata}]")
            


# # def main():
# #     # st.set_page_config("Chat PDF")
# #     st.header("Chat with PDF using Gemini💁")

# #     user_question = st.text_input("Ask a Question from the PDF Files")

# #     if user_question:
# #         user_input(user_question)

# #     with st.sidebar:
# #         # st.title("Menu:")
# #         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
# #         if st.button("Submit & Process"):
# #             with st.spinner("Processing..."):
# #                 raw_text = get_pdf_text(pdf_docs)
# #                 text_chunks = get_text_chunks(raw_text)
# #                 get_vector_store(text_chunks)
# #                 st.success("Done")


# # if __name__ == "__main__":
# #     main()
    
