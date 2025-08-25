import streamlit as st
import pandas as pd
from datetime import datetime

# Import your existing chatbot components
from main import (  # Replace with your actual filename
    config, data_manager, vector_manager, 
    external_tools, conversation_manager, chatbot
)

# Page configuration
st.set_page_config(
    page_title="E-commerce Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "customer_id" not in st.session_state:
    st.session_state.customer_id = ""

if "location" not in st.session_state:
    st.session_state.location = ""

# Sidebar for settings
with st.sidebar:
    st.header("Chat Settings")
    st.session_state.customer_id = st.text_input(
        "Customer ID", 
        value=st.session_state.customer_id,
        help="Enter customer ID for personalized recommendations"
    )
    
    st.session_state.location = st.text_input(
        "Location", 
        value=st.session_state.location,
        help="Enter location for weather-based recommendations"
    )
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.title("🤖 E-commerce Shopping Assistant")
st.write("Welcome! How can I help you find the perfect products today?")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What are you looking for today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get chatbot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chatbot.process_message(
                customer_id=st.session_state.customer_id,
                message=prompt,
                location=st.session_state.location
            )
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Product search feature in sidebar
with st.sidebar:
    st.divider()
    st.header("Product Search")
    search_query = st.text_input("Search products by description")
    if search_query:
        results = vector_manager.search_products(search_query)
        if results:
            st.write("Found products:")
            for i, doc in enumerate(results):
                with st.expander(f"Product {i+1}: {doc.metadata['name']}"):
                    st.write(f"Price: ${doc.metadata['price']}")
                    st.write(f"Category: {doc.metadata['category']}")
                    st.write(doc.page_content)
        else:
            st.write("No products found matching your search")

# Customer information display
if st.session_state.customer_id:
    with st.sidebar:
        st.divider()
        st.header("Customer Info")
        customer_data = data_manager.get_customer_data(st.session_state.customer_id)
        if customer_data:
            st.json(customer_data)
        else:
            st.write("No customer data found")