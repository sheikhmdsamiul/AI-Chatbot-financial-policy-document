import streamlit as st
from modules.ragchain import rag_chat
from modules.extract_pdf import vectorstore

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# title
st.title("Financial Policy Document Q&A")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the document"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # user input message in chat
    with st.chat_message("user"):
        st.write(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # Get response from RAG chat function
            response = rag_chat(
                query=prompt,
                chat_history=st.session_state.chat_history,
                vector_store=vectorstore
            )
            st.write(response)
    
    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Update chat history
    st.session_state.chat_history.extend([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ])