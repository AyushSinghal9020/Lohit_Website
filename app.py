__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st 
import os 

import chromadb

from langchain_chroma import Chroma
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage
from langchain_community.embeddings import HuggingFaceEmbeddings

file = open('data.json').read() 
data = file.split('\n')

client = chromadb.PersistentClient()
embedding_function = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')

try : collection = client.create_collection(name = 'extracted_data')

except : 
    
    client.delete_collection(name = 'extracted_data')
    collection = client.create_collection(name = 'extracted_data')

collection.add(
    ids = [str(index) for index in range(len(data))] , 
    documents = [doc for doc in data]
)

vc = Chroma(
    persist_directory = 'chroma_db' , 
    client = client , 
    collection_name = 'extracted_data' , 
    embedding_function = embedding_function
)

def search(query) : 

    prompt = open('prompt.txt').read()
    similar_docs = vc.similarity_search_with_score(query = prompt , k = len(vc.get()['documents']))
    similar_docs = sorted(similar_docs , key = lambda val : val[1] , reverse = True)

    img_doc = []
    text_doc = []

    num_img_slides = 5
    num_slides = 5


    for doc in similar_docs : 

        if len(img_doc) >= num_img_slides and len(text_doc) >= num_slides : break 

        content = doc[0].page_content

        if 'image' in content : 
            if len(img_doc) < num_img_slides : img_doc.append(content)
        else : 
            if len(text_doc) < num_slides : text_doc.append(content)

    img_links_with_captions = '\n'.join(img_doc)
    context = '\n'.join(text_doc)

    prompt = prompt.format(
        img_links_with_captions , 
        context , 
        query
    )

    chat = ChatCohere(cohere_api_key = 'FELFXgLGfcqsy4eh4Q75dXNT7VyIQjKZmhkiIug3')
    messages = [HumanMessage(content = prompt)]
    response = chat.invoke(messages)

    return response.content



def check_prompt(prompt) : 

    try : 

        prompt.replace('' , '')
        return True 
    except : return False


def check_mesaage() : 
    '''
    Function to check the messages
    '''

    if 'messages' not in st.session_state : st.session_state.messages = []

check_mesaage()

for message in st.session_state.messages : 

    with st.chat_message(message['role']) : st.markdown(message['content'])

prompt = st.chat_input('Ask me anything')

if check_prompt(prompt) :

    with st.chat_message('user'): st.markdown(prompt)

    st.session_state.messages.append({
        'role' : 'user' , 
        'content' : prompt
    })

    if prompt != None or prompt != '' : 

        response = search(prompt)

        with st.chat_message('assistant') : st.markdown(response)

        st.session_state.messages.append({
            'role' : 'assistant' , 
            'content' : response
        })
