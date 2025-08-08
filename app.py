import streamlit as st 
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



load_dotenv() #Carrego a minha api key 
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY') #Defino a variável de ambiente OPENAI_API_KEY com o valor carregado do .env
    
persist_directory = 'dbv' #Crio um banco de dados vetorial

embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large' ) #Modelo de embeddings usado para converter texto em vetores semânticos para busca


def path_doc(file): 
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tf: #Função onde converto um arquivo em binário para um arquivo temporario no disco local

        tf.write(file.read())
        path_archive = tf.name

    try:    

        documents = PyPDFLoader(file_path=path_archive).load() #Função para carregar esse arquivo que está no meu disco local

        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 500)#Determino a separação do texto em conjuntos

        chunks = text_splitter.split_documents(documents=documents)#Uso a função split_documents para dividir os textos 

        return chunks #retorno o chunk(texto) dividido
    
    finally:
        if os.path.exists(path_archive): #Verifico se o arquivo criado no meu disco local existe e removo ele depois de ter separado em chunks
            os.remove(path_archive)


def load_existing_vector_store(): #Função responsável de verificar se um vector_store (banco de dados vetorial) já existi ou não

    if os.path.exists(os.path.join(persist_directory)): #Verifico se existe algum endereço de caminho para o meu vector_store
        vector_store = Chroma(embedding_function=embeddings, persist_directory=persist_directory) #Faço a conexão
        return vector_store

    else:
        return None #Se não encontrar, retorna um None

def add_to_vector_store(chunks, vector_store = None): #Função que adiciona os novos chunks para o vector_store, caso ele não exista ele cria um novo

    if vector_store: #Se existir, adiciona os novos chunks
        
        vector_store.add_documents(chunks)

    else: #Se não existir, crio um do zero passando os chunks, embeddings e o meu banco de dados vetorial
        
        vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)

    return vector_store #Retorna o novo vector_store


def respond_to_ask_ai(llm, query, vector_store): #Função do modelo llm para buscar a resposta dentro do vector_store e responder o user


    llm_models = ChatOpenAI(model=llm) #Passo o modelo escolhido pelo user
    retriever = vector_store.as_retriever() #Função que faz a busca do vetores dentro do meu persist_directory (banco de dados vetorial)

    #Sistema de prompt para ser utilizado pelo o modelo llm
    system_prompt = '''
    
    I want you to be a bidding assistant, searching for data within .pdf files (notices) provided to you.

    Furthermore, I want your response to be in Markdown format and based on the data extracted from the .pdf file provided by the user.
    
    Context: {context}
'''
    messages = [('system', system_prompt)] #Lista de mensagens, passando em primeiro o prompt do sistema para ser executado

    for message in st.session_state.messages: #Faço um for no meu banco de dados pelo streamlit onde busco perguntas (human) e resposta(ai)

        messages.append((message.get('role'), message.get('content'))) #Adiciono na minha lista em tuplas

    messages.append(('human', '{input}')) #Adiciono um placeholder de input do usuário que será preenchido no prompt final

    prompt = ChatPromptTemplate.from_messages(messages) #Utilizo um modelo de Prompt do Langchain para passar a minha lista de mensagens

    doc_chains = create_stuff_documents_chain(llm=llm_models, prompt=prompt) #Cria uma cadeia (chain) que insere os documentos diretamente no prompt antes da resposta do LLM (modo 'stuff')

    retrieval_chains = create_retrieval_chain(retriever=retriever, combine_docs_chain=doc_chains) #Cria uma cadeia completa de recuperação: busca documentos relevantes e insere eles no prompt para gerar a resposta com base no conteúdo dos PDFs.



    response = retrieval_chains.invoke({'input': query}) #Invoco a resposta encontrada pelo meu modelo llm dentro do meu persist_directory

    return response.get('answer') #Retorna apenas a resposta


vector_store = load_existing_vector_store() #Crio uma variavel para receber a minha função de carregar o meu vector_store, caso exista

st.set_page_config(page_title='ChatBot', page_icon='🤖⚙') #Inicio a pagina do chatbot com o Streamlit

st.title('CHATBOT') #Pge Title

with st.sidebar: #Crio um SideBar na pagina, menu lateral
    st.text(body='Bem-Vindo ao ChatBot UPDOC, um ChatBot responsável por tratar qualquer docs em dados e responder' \
    ' a sua pergunta com base nos dados do documento que você fez o upload! 📄🚀') #Explicação do funcionamento do chatbot

    st.warning('Será aceito apenas documentos formato PDF!') #Aviso

    upload = st.file_uploader(label='Docs em formato em PDF', type=['pdf'], accept_multiple_files=True) #Função do Streamlit que recebe multiplos arquivos
    
    if upload: #Verifico se teve algum documento subido
     with st.spinner('Processando Documento...'): #Função do Streamlit de aviso de carregamento
        all_chunks = [] #Lista vazia
        for up in upload: #Passo pela variavel 'upload' um por um para ver quantos arquivos o user subiu
            chunks = path_doc(file=up) #Cada arquivo subido vai passar na minha função 'path_doc' para converter de binario para endereço local 
            all_chunks.extend(chunks) #A cada arquivo convertido para endereço local, eu extendo na minha lista chamada 'all_chunks'

        vector_store = add_to_vector_store(chunks=all_chunks, vector_store=vector_store) #Crio uma variavel, onde recebe minha função 'add_to_vector_store'

    models_llm = st.selectbox(label = 'Escolha um modelo de linguagem', options=['gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4o']) #Dentro do sidebar ainda crio uma caixa de seleção de modelos llm
        
        

question = st.chat_input(placeholder='Pergunte alguma coisa') #Input do user

if 'messages' not in st.session_state: #Inicio o meu session_state
    st.session_state['messages'] = []
            
if question and vector_store: #Se tiver pergunta do user e novos chunks para serem adicionados no vector_store

    for message in st.session_state.messages: #Verifico todas as mensagens dentro do meu session_state, uma por uma

        st.chat_message(message.get('role')).write(message.get('content')) #O chat_message do streamlit tem função de mostrar na page o ator da mensagem e o conteudo, então eu faço um get em no session
    st.chat_message('human').write(question) #Mostro novamente o ator human e a nova pergunta dele para o chatbot
    st.session_state.messages.append({'role': 'human', 'content': question}) #Aqui eu adiciono a ultima pergunta feita pelo human para dentro do meu session_state

    with st.spinner(text='Carregando Resposta'): #Ilustro o carregamento da resposta do chatbot
     response = respond_to_ask_ai(llm = models_llm, query = question, vector_store = vector_store) #Função do chatbot para processar a pergunta e buscar a resposta dentro do meu persist_directory

     st.chat_message('ai').write(response) #Com o chat_message mostro o ator que é o chatbot e a resposta dele na page principal
     st.session_state.messages.append({'role': 'ai', 'content': response}) #Depois eu salvo a ultima resposta do chatbot para dentro do meu session_state
        

            