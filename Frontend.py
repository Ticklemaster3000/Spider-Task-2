import streamlit as st
from API import ask  # import your custom ask function

# Set the page title
st.title('Custom Chat Model')

# Initialize session message history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'texts' not in st.session_state:
    st.session_state['texts'] = []

# sidebar for displaying documents and their similarity scores
st.sidebar.title("Relevant Matches with Similarity Scores (Euclidean Distance)")

for item in st.session_state['texts']:
    title = item['title']

    st.sidebar.markdown(f"**{title}**")
    st.sidebar.caption(f"Similarity: {item['score']}")

    # Unique key to avoid collisions
    if st.sidebar.button("Read more", key=f"read_{title}"):
        st.session_state[f"show_full_{title}"] = True
        st.rerun()

    if st.session_state.get(f"show_full_{title}", False):
        with st.expander(f"📖 Full Text - {title}", expanded=True):
            st.write(item['full'])
            if st.button("Close", key=f"close_{title}"):
                st.session_state[f"show_full_{title}"] = False
                st.rerun()

# Display previous messages
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Handle new user input
if prompt := st.chat_input("Enter your query"):
    # Show user message
    st.session_state['messages'].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and show assistant's response using custom model
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask(prompt)
            st.session_state["texts"] = response[1]
        st.markdown(response[0])

    st.session_state['messages'].append({"role": "assistant", "content": response[0]})
    st.rerun()

