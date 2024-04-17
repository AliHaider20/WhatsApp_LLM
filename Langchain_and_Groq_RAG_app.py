import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader, WhatsAppChatLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

## load the Groq API key
groq_api_key=os.getenv('GROQ_API_KEY')

# ------------------------------------- Streamlit App -------------------------------------           
# if "vector" not in st.session_state:
#     st.session_state.embeddings=OllamaEmbeddings(model = "all-minilm", top_k=3)
#     # st.session_state.loader=WebBaseLoader("https://docs.smith.langchain.com/")
#     st.session_state.loader=WhatsAppChatLoader("WhatsApp Chat with Family No 1.txt")
#     st.session_state.docs=st.session_state.loader.load()

#     st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
#     st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs )
#     st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

# st.title("ChatGroq Demo")
# llm=ChatGroq(groq_api_key=groq_api_key,
#              model_name="mixtral-8x7b-32768")

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
# document_chain = create_stuff_documents_chain(llm, prompt)
# retriever = st.session_state.vectors.as_retriever()
# retrieval_chain = create_retrieval_chain(retriever, document_chain)

# prompt=st.text_input("Input you prompt here")

# if prompt:
#     start=time.process_time()
#     response=retrieval_chain.invoke({"input":prompt})
#     print("Response time :",time.process_time()-start)
#     st.write(response['answer'])

#     # With a streamlit expander
#     with st.expander("Document Similarity Search"):
#         # Find the relevant chunks
#         for i, doc in enumerate(response["context"]):
#             st.write(doc.page_content)
#             st.write("--------------------------------")
    

# ------------------------------------------------- Gradio App -------------------------------------------------

import gradio as gr
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Initialize the Ollama Embeddings and other necessary components
embeddings = OllamaEmbeddings(model="all-minilm", top_k=1)
loader = WhatsAppChatLoader("WhatsApp Chat with Family No 1.txt")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(docs)
vectors = FAISS.from_documents(final_documents, embeddings)
retriever = vectors.as_retriever()

# Initialize the ChatGroq model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

prompt= ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question:{input}

    """
)

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# rag_chain = (
#     {"context": retriever | format_docs, "input": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# Define the Gradio interface
def chatgroq_demo(ques):
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input": ques})
    # response_time = time.process_time() - start

    # Prepare the response
    context = "\n".join([doc.page_content for doc in response["context"]])
    answer = response

    return f"Answer:\n{answer}"

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        bot_message = chatgroq_demo(message)
        chat_history.append((message, bot_message))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(share=True)
# import gradio as gr

# def upload_file(files):
#     file_paths = [file.name for file in files]
#     print(open(files[0], encoding='latin').read()[:100])
#     return file_paths

# with gr.Blocks() as demo:
#     file_output = gr.File()
#     upload_button = gr.UploadButton("Click to Upload a File", file_types=["txt"], file_count="single")
#     upload_button.upload(upload_file, upload_button, file_output)



# demo.launch()

	