import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from PIL import Image
from model import model_prediction
from base64 import b64encode
from io import BytesIO
import base64

from rag.chain import build_rag_chain

load_dotenv()

# Page layout
st.set_page_config(page_title="plant chatbot",
                   layout='wide')

# Custom CSS to move the subheader upward
st.markdown("""
    <h3 style='margin-top: -50px; font-size: 22px; '>Plant Disease Chatbot Expert</h3>
""", unsafe_allow_html=True)

# Loading Groq API Key & LLM model
llm=ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-8b-8192",
    temperature=0.6
)

# Adding rag chain
rag_chain=build_rag_chain(llm)

# System Message
system_message=SystemMessage(content="""
    Hi there! 👋 I'm AgroBot 8, your friendly agricultural assistant 🤖🌿. I always have friendly and professional responses.
    I was proudly created by Group 8, a team of Data Science and Machine Learning students from Refactory Academy, Cohort 1 (2025 Class) in Uganda. 🎓
    I specialize in diagnosing plant diseases and offering expert, farmer-friendly advice. 
    You can chat with me about any agriculture-related topics from crop health to pest management.
    I like to chat while using emojis depending on the suitiation. 
    However, I do not use emojis at the begining of a sentence.
    I am also built using a trained ML model. When my users upload leaf images, I scan through and tell them the plant condition.
    When I’m given the name of a plant disease (e.g., *tomato blight*), I will respond with:
    - A brief overview of the disease
    - Key symptoms to look out for
    - Recommended treatments or pesticides
    - Preventive tips to keep your crops healthy
    I’ll keep my answers simple, actionable, and under 300 words so they’re easy for any farmer to understand.
    Let’s grow smarter together! 🌱
    The entire response I return is less than 300 words. Brief and straight to the point.
    When concluding, I like to ask users for more questions or opinions to keep the conversation going.
""")

# chat history uisng session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"]=[]

# Texts and image upload
text = st.chat_input("Type your message here...")

# Image Upload and Display
with st.sidebar:
    st.header("Files")
    test_image = st.file_uploader("Upload an Image", type=["jpg", "png"])


# Generating responses
def generate_response(chat_history):
    latest_question = chat_history[-1].content if hasattr(chat_history[-1], "content") else chat_history[-1]["user"]
    response = rag_chain.invoke({"question": latest_question})
    return response.content if hasattr(response, "content") else response
# Function to get chat history
def get_history():
    messages = [system_message]
    for chat in st.session_state["chat_history"]:
        if chat.get("user"):
            messages.append(HumanMessage(content=chat["user"]))
        if chat.get("assistant"):
            messages.append(AIMessage(content=chat["assistant"]))
    return messages
  
# Human message response
if text:
    with st.spinner("Thinking...."):
        prompt=HumanMessage(content=text)
        chat_history=get_history()
        chat_history.append(prompt)
        response=generate_response(chat_history)
        st.session_state["chat_history"].append({
            'user':text,
            "assistant":response})

# Ensures there is a new image not the one already processed
if "last_uploaded_image" not in st.session_state:
    st.session_state["last_uploaded_image"] = None
    
# --- Image Upload Logic ---
if test_image is not None and test_image != st.session_state["last_uploaded_image"]:
    with st.spinner("Analyzing image and generating advice..."):
        
        # Convert uploaded image to base64
        img = Image.open(test_image)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        img_md = f'<img src="data:image/png;base64,{img_base64}" width="300"/>'
        
        # Use the model to predict
        prediction_index = model_prediction(test_image)
        labels = []
        with open("labels.txt") as f:
            content = f.readlines()
            for line in content:
                labels.append(line.strip())
        predicted_disease = labels[prediction_index]

        # Generate expert advice from LLM based on prediction
        prompt_text = f"The detected plant disease is {predicted_disease}. Provide expert advice."
        prompt = HumanMessage(content=prompt_text)
        chat_history = get_history()
        chat_history.append(prompt)
        
        # expert Advise
        def get_expert_advice(disease_name: str) -> str:
            prompt = f"The detected plant disease is {disease_name}. Provide expert advice."
            result = rag_chain.invoke({"question": prompt})
            return result.content if hasattr(result, "content") else result
        
        expert_response = get_expert_advice(predicted_disease)


        # Append AI message with prediction + expert advice
        ai_message = f"{img_md}<br><br> The detected disease is: {predicted_disease}\n\n{expert_response}"
        st.session_state["chat_history"].append({
            "user": "You uploaded a plant image for diagnosis.", 
            "assistant": f"The detected disease is: {predicted_disease}\n\n{expert_response}",
            "image": img_md
        })
        st.session_state["last_uploaded_image"] = test_image

        
# CSS for chat display alignment
st.markdown(
    """
    <style>
    .user-message {
    padding: 10px;
    border-radius: 10px;
    max-width: 70%;
    margin-left: auto;
    margin-right: 200px;
    text-align: right;
}

.bot-message {
    padding: 10px;
    border-radius: 10px;
    max-width: 70%;
    margin-right: auto;
    margin-left: 10px;
    text-align: left;
}
    </style>
    """, unsafe_allow_html=True)

# display messages
for chat in st.session_state['chat_history']:
    user_msg = chat.get('user', '')
    assistant_msg = chat.get('assistant', '')
    img_html = chat.get('image', None)
    st.markdown(f"<div class='user-message'>👱‍♀️ {user_msg}</div>", unsafe_allow_html=True)
    
    if img_html is not None:
        # Show image just before assistant message
        st.markdown(img_html, unsafe_allow_html=True)
    
    st.markdown(f"<div class='bot-message'>🤖 {assistant_msg}</div>", unsafe_allow_html=True)
    st.markdown("---")

    
# Side Bar for image uploading   
with st.sidebar:
    # Conversation history Summary (Questions asked)
    st.subheader("🕓 Conversation History")
    if st.session_state["chat_history"]:
        for i, chat in enumerate(reversed(st.session_state["chat_history"])):
            if isinstance(chat, dict) and "user" in chat:
                st.markdown(f"**{len(st.session_state['chat_history']) - i}.** {chat['user']}")
    else:
        st.write("No messages yet.")


        
