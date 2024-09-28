# # import streamlit as st
# # import os
# # from langchain_groq import ChatGroq
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain.chains.combine_documents import create_stuff_documents_chain
# # from langchain_core.prompts import ChatPromptTemplate
# # from langchain.chains import create_retrieval_chain
# # from langchain_community.vectorstores import FAISS
# # from langchain_community.document_loaders import PyPDFDirectoryLoader
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # from dotenv import load_dotenv
# # import os
# # load_dotenv()

# # ## load the GROQ And OpenAI API KEY 
# # groq_api_key=os.getenv('GROQ_API_KEY')
# # os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

# # # st.title("Gemma Model Document Q&A")

# # llm=ChatGroq(groq_api_key=groq_api_key,
# #              model_name="Llama3-8b-8192")

# # prompt=ChatPromptTemplate.from_template(
# # """
# # Answer the questions based on the provided context only.
# # Please provide the most accurate response based on the question
# # <context>
# # {context}
# # <context>
# # Questions:{input}

# # """
# # )

# # def vector_embedding():

# #     if "vectors" not in st.session_state:

# #         st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
# #         st.session_state.loader=PyPDFDirectoryLoader("./us_census") ## Data Ingestion
# #         st.session_state.docs=st.session_state.loader.load() ## Document Loading
# #         st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
# #         st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #splitting
# #         st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings


# # prompt1=st.text_input("Enter Your Question From Doduments")


# # if st.button("Documents Embedding"):
# #     vector_embedding()
# #     st.write("Vector Store DB Is Ready")

# # import time

# # if prompt1:
# #     document_chain=create_stuff_documents_chain(llm,prompt)
# #     retriever=st.session_state.vectors.as_retriever()
# #     retrieval_chain=create_retrieval_chain(retriever,document_chain)
# #     start=time.process_time()
# #     response=retrieval_chain.invoke({'input':prompt1})
# #     print("Response time :",time.process_time()-start)
# #     st.write(response['answer'])

# #     # With a streamlit expander
# #     with st.expander("Document Similarity Search"):
# #         # Find the relevant chunks
# #         for i, doc in enumerate(response["context"]):
# #             st.write(doc.page_content)
# #             st.write("--------------------------------")


# import streamlit as st
# import os
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# import os
# load_dotenv()


# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import CharacterTextSplitter
# # from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains import RetrievalQA

# ## load the GROQ And OpenAI API KEY 
# groq_api_key=os.getenv('GROQ_API_KEY')
# os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

# # st.title("Gemma Model Document Q&A")

# loader = PyPDFLoader("MLDOC (1).txt")
# documents = loader.load()

# # Splitting the data into chunk
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
# docs = text_splitter.split_documents(documents=documents)
     
# llm=ChatGroq(groq_api_key=groq_api_key,
#              model_name="gemini-1.5-pro")

# prompt=ChatPromptTemplate.from_template(
# """
# Answer the questions based on the provided context only.
# Please provide the most accurate response based on the question
# <context>
# {context}
# <context>
# Questions:{input}

# """
# )

# def vector_embedding():

#     if "vectors" not in st.session_state:

#         embedding=st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
#         st.session_state.loader=PyPDFDirectoryLoader(docs) ## Data Ingestion
#         data = st.session_state.docs=st.session_state.loader.load() ## Document Loading
#         st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=10) ## Chunk Creation
#         final_doc=st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #splitting
#         # index=FAISS.IndexFlatL2(5)
#         # index.add(data)
#         # st.session_state.vectors=FAISS.from_documents(final_doc, embedding,index) 
#         vectorstore = FAISS.from_documents(final_doc, embedding)
#         vectorstore.save_local("faiss_index_")
#         persisted_vectorstore = FAISS.load_local("faiss_index_",embedding,allow_dangerous_deserialization=True)
#         retriever = persisted_vectorstore.as_retriever()
        
#         qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        
#         while True:
#             query = input("Type your query if you want to exit type Exit: \n")
#             if query == "Exit":
#                 break
            
  
#         result = qa.run(query)
#         print(result)
    

# prompt1=st.text_input("Enter Your Question From Doduments")

# if st.button("Documents Embedding"):
#     vector_embedding()
#     st.write("Vector Store DB Is Ready")

# import time

# if prompt1:
#     document_chain=create_stuff_documents_chain(llm,prompt)
#     retriever=st.session_state.vectors.as_retriever()
#     retrieval_chain=create_retrieval_chain(retriever,document_chain)
#     start=time.process_time()
#     response=retrieval_chain.invoke({'input':prompt1})
#     print("Response time :",time.process_time()-start)
#     st.write(response['answer'])

#     # With a streamlit expander
#     with st.expander("Document Similarity Search"):
#         # Find the relevant chunks
#         for i, doc in enumerate(response["context"]):
#             st.write(doc.page_content)
#             st.write("--------------------------------")



# from langchain_astradb import AstraDBVectorStore
# # from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# import os
# import pandas as pd
# # from ecommbot.data_converter import dataconveter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings


# load_dotenv()

# # OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
# ASTRA_DB_API_ENDPOINT=os.getenv("ASTRA_DB_API_ENDPOINT")
# ASTRA_DB_APPLICATION_TOKEN=os.getenv("ASTRA_DB_APPLICATION_TOKEN")
# ASTRA_DB_KEYSPACE=os.getenv("ASTRA_DB_KEYSPACE")

# # embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
# embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# def ingestdata(status):
#     vstore = AstraDBVectorStore(
#             embedding=embedding,
#             collection_name="ecomerce",
#             api_endpoint=ASTRA_DB_API_ENDPOINT,
#             token=ASTRA_DB_APPLICATION_TOKEN,
#             namespace=ASTRA_DB_KEYSPACE,
#         )
    
#     storage=status
    
#     if storage==None:
#         # docs=dataconveter()
#         docs=
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



