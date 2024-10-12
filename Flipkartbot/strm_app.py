import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from Flipkartbot.data_ingestion import data_ingestion
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set up the model
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.5)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def generation(vstore):
    retriever = vstore.as_retriever(search_kwargs={"k": 3})
    
    retriever_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history,"
        "formulate a standalone question which can be understood without the chat history."
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", retriever_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)
    
    PRODUCT_BOT_TEMPLATE = """
    Your ecommercebot bot is an expert in product recommendations and customer queries.
    It analyzes product titles and reviews to provide accurate and helpful responses.
    Ensure your answers are relevant to the product context and refrain from straying off-topic.
    Your responses should be concise and informative.
    
    CONTEXT:
    {context}
    
    QUESTION: {input}
    
    YOUR ANSWER:
    """
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PRODUCT_BOT_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    return conversational_rag_chain

# Streamlit UI setup
st.set_page_config(page_title="E-commerce Bot", page_icon="🛒", layout="wide")

# Custom CSS to style the app
st.markdown("""
<style>
.stApp {
    background-color: #1E1E1E;
    color: white;
}
.user-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background-color: #4CAF50;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
}
.bot-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background-color: #2196F3;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
}
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: flex-start;
}
.chat-message.user {
    background-color: #4CAF50;
    margin-left: 20%;
}
.chat-message.bot {
    background-color: #2196F3;
    margin-right: 20%;
}
.chat-bubble {
    flex: 1;
    padding-left: 1rem;
}
.timestamp {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.7);
    text-align: right;
}
.stTextInput > div > div > input {
    background-color: #2B2B2B;
    color: white;
}
.stButton > button {
    background-color: #4CAF50;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Initialize vstore and conversational_rag_chain
if 'vstore' not in st.session_state:
    st.session_state.vstore = data_ingestion("done")

if 'conversational_rag_chain' not in st.session_state:
    st.session_state.conversational_rag_chain = generation(st.session_state.vstore)

# Chat interface
st.title("E-commerce Assistant")

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user">
            <div class="user-avatar">Hi</div>
            <div class="chat-bubble">
                {message["content"]}
                <div class="timestamp">{message["timestamp"]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot">
            <div class="bot-avatar">🛒</div>
            <div class="chat-bubble">
                {message["content"]}
                <div class="timestamp">{message["timestamp"]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# User input
with st.container():
    user_input = st.text_input("Type your message here...", key="user_input")
    send_button = st.button("Send")

if send_button and user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input, "timestamp": "Just now"})
    
    # Generate bot response
    with st.spinner("Thinking..."):
        response = st.session_state.conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id": "streamlit_user"}
            },
        )["answer"]
    
    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response, "timestamp": "Just now"})
    
    # Rerun the app to update the chat display
    st.experimental_rerun()