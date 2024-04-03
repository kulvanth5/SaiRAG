import streamlit as st
import torch

from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.http import models
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

client = QdrantClient(url="https://ccaa5582-c81c-49c3-93a9-866c4617ce0e.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="S1RryvNzKIXBwLYriv2FVqj11oUgoyrrxyLJDNfiyi5he7b6SRs-jg",)
modelst = SentenceTransformer("msmarco-distilbert-dot-v5")

def user_search(qn):
    
    hits = client.search(
        collection_name="database",
        query_vector=modelst.encode(qn).tolist(),
        limit=5,
    #     with_vectors=True,
        search_params=models.SearchParams(hnsw_ef=128,exact=True),
    )

    res = []
    
    st.write("Top matching results !!\n")
    
    for hit in hits:
        
        if hit.score > 77:
            res.append(hit.payload)
            
        st.write(hit.payload)
        st.write('matching score = ',hit.score)
        
    return res

config = BitsAndBytesConfig(
    load_in_4bit=True, #changing the model weights to bit floating points
    bnb_4bit_compute_dtype=torch.float16, #the computational matrices that will be used for training or inference in floating point 16
    bnb_4bit_quant_type="nf4", #this is the quantization type
    bnb_4bit_use_double_quant=True, #doing double quantization
)

messages = []

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1",quantization_config=config,device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("/mistralai/Mistral-7B-v0.1")

def generate(context,query):
    
    print('entered for generation')
    global messages 
        
    prompt = f"""
    Context information is below.
    ---------------------
    Chat History:
    {messages}
    Context:
    {context}
    ---------------------
    Given the context and chat history, answer the query.
    Query: {query}
    Instructions:
         - Generate an effective response by leveraging the context effectively to address the query cohesively, ensuring the information provided is relevant and coherent within the given context. 
        - Leverage the usage of transliterations, anecdotes or short stories from the context to answer the query.
        - When the context is "None," utilize the chat history as a reference to guide the generation of new text, focusing on creating original content while avoiding direct duplication of previous dialogues.
        - Do not stop the text generation abruptly in the middle of a sentence. Ensure that the reponse ends with good meaning and coherence.
        - Once in a while, greet the user with 'Sairam!' or 'Embodiment of Divine Love!!' to establish a friendly tone.
        - Embody the sacred duty of a dedicated messenger, entrusted with the profound responsibility of imparting the timeless teachings and profound wisdom of Bhagawan Sri Sathya Sai Baba with unwavering devotion and reverence.
        - Please refrain from incorporating personal knowledge into your response. If the context provided doesn't suffice or isn't aligned with the query, kindly conclude by using, "In my continuous learning journey, I strive to provide accurate responses. However, if the context is unclear, I may not be able to provide a relevant answer. Please consider refining your query for better assistance."
        - Sometimes, wish the user better luck in their journey towards pursuing spirituality. 


    Answer:
    """
    
    encodeds = tokenizer(prompt, return_tensors="pt")
    model_inputs = encodeds.to("cuda")

    generated_ids = model.generate(
        **model_inputs,
        num_beams=3,
        max_new_tokens=4096,
        do_sample=True,
        early_stopping=True,
        repetition_penalty = 2.02,
    )

    decoded_text = tokenizer.batch_decode(generated_ids)
    i = decoded_text[0].find('Answer:') + 7
    
    messages.append('User:'+query)
    messages.append('Assistant:'+decoded_text[0][i:])
    
    if len(messages) > 5:
        messages.pop(0)
        
    print('\nlength of generated text is ',len(decoded_text[0][i:]),'\n\n')
    
    st.write(f"Model generated answer\n{decoded_text[0][i:]}")

def check_opinion(query):
    
    phrases = ["you","yours","yourself"," u "," urs "," urself ","your"," ur "]
    
    for phrase in phrases:
        
        if phrase in query:
            print('returning true')
            return True
        
    return False

st.title("Chatbot on Sai Literature")

query = st.text_input("Ask your question :")
btn = st.button('Search')
                
if btn:

    try:
         context = None
    
        if not check_opinion(query):
            context = user_search(query)
    
        generated_text = generate(context,query)

else:
    st.warning("please ask some question to get an answer !!")

