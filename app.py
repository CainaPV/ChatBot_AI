import streamlit as st 
import time
from dotenv import load_dotenv
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    
persist_directory = 'dbv'

embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large' )


def path_doc(file):
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tf:

        tf.write(file.read())
        path_archive = tf.name

    try:    

        documents = PyPDFLoader(file_path=path_archive).load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 500)

        chunks = text_splitter.split_documents(documents=documents)

        return chunks
    
    finally:
        if os.path.exists(path_archive):
            os.remove(path_archive)


def load_existing_vector_store():

    if os.path.exists(os.path.join(persist_directory)):
        vector_store = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

    else:
        return None

def add_to_vector_store(chunks, vector_store = None):

    if vector_store:
        
        vector_store.add_documents(chunks)

    else:
        
        vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)

    return vector_store


def respond_to_ask_ai(llm, query, vector_store):


    llm = ChatOpenAI(model=models_llm)
    retriever = vector_store.as_retriever()

    system_prompt = '''
    
    I want you to be a bidding assistant, searching for data within .pdf files (notices) provided to you.

    Furthermore, I want your response to be in Markdown format and based on the data extracted from the .pdf file provided by the user.
    
    Context: {context}
'''
    messages = [('system', system_prompt)] 

    for message in st.session_state.messages:

        messages.append((message.get('role'), message.get('content')))

    messages.append(('human', '{input}'))

    prompt = ChatPromptTemplate.from_messages(messages)

    doc_chains = create_stuff_documents_chain(llm=llm, prompt=prompt)

    retrieval_chains = create_retrieval_chain(retriever=retriever, combine_docs_chain=doc_chains)

    response = retrieval_chains.invoke({'input': query})

    return response.get('answer')


vector_store = load_existing_vector_store()

st.set_page_config(page_title='ChatBot', page_icon='🤖⚙')

st.title('CHATBOT')

with st.sidebar:
    st.text(body='Bem-Vindo ao ChatBot UPDOC, um ChatBot responsável por tratar qualquer docs em dados e responder' \
    ' a sua pergunta com base nos dados do documento que você fez o upload! 📄🚀')

    st.warning('Será aceito apenas documentos formato PDF!')

    upload = st.file_uploader(label='Docs em formato em PDF', type=['pdf'], accept_multiple_files=True)
    
    if upload:
     with st.spinner('Processando Documento...'):
        all_chunks = []
        for up in upload:
            chunks = path_doc(file=up)
            all_chunks.extend(chunks)

        vector_store = add_to_vector_store(chunks=all_chunks, vector_store=vector_store)

    models_llm = st.selectbox(label = 'Escolha um modelo de linguagem', options=['gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4o'])
        
        

question = st.chat_input(placeholder='Pergunte alguma coisa')

if 'messages' not in st.session_state:
    st.session_state['messages'] = []
            
if question and vector_store:

    for message in st.session_state.messages:

        st.chat_message(message.get('role')).write(message.get('content'))
    st.chat_message('human').write(question)
    st.session_state.messages.append({'role': 'human', 'content': question})

    with st.spinner(text='Carregando Resposta'):
     response = respond_to_ask_ai(llm = models_llm, query = question, vector_store = vector_store)

     st.chat_message('ai').write(response)
     st.session_state.messages.append({'role': 'ai', 'content': response})
        

            