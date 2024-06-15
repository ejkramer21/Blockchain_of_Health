import streamlit as st
from summarizer import Summarize
from NER_with_streamlit import NER

st.markdown("<h1 style='text-align: center;'>Blockchain of Health</h1>",unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

col1, spacer, col2 = st.columns([1, 0.5, 1])
if uploaded_file is not None:
    sum = Summarize("C:/Users/erinj/Downloads/erin.pdf")
    ner = NER(uploaded_file)
    with col2:  
        st.markdown("<h3> Summary of the document</h3>",unsafe_allow_html=True)
        sum.callRag()
    with col1:
        st.markdown("<h3>NER Model Output from PDF</h3>",unsafe_allow_html=True)
        ner.runNER()
    