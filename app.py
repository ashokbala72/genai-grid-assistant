import os
os.environ["OPENAI_API_KEY"] = "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"


import streamlit as st
from retriever_chain import build_chain

st.set_page_config(page_title="⚡ Grid Event Assistant", layout="wide")
st.title("⚡ Real-Time Grid Event Explainer")

query = st.text_input("Ask about recent grid events (e.g., 'What happened in Substation-12?'):")

if query:
    qa_chain = build_chain()
    result = qa_chain.run(query)
    st.write("🧠 AI Response:")
    st.success(result)
