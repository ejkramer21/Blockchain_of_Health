import streamlit as st
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline

class Summarize:
    file=""
    loaders={}
    summary=""
    def __init__(self,file):
        self.file=file
        self.loaders["txt"]=TextLoader
        self.loaders["pdf"]=PyPDFLoader
        self.loaders["csv"]=CSVLoader
        
    def getLoader(self):
        file_type = str(self.file).split('.')[1]
        file_type = file_type.split("'")[0]
        st.markdown(file_type)
        loader = self.loaders[file_type]
        load = loader(str(self.file))
        data = load.load()
        return data
        
    def callRag(self):
        st.markdown("Processing your file...")
        data = self.getLoader()
        st.markdown("Data Loaded")
        # for d in data:
        #     st.markdown(d.page_content.split("\n"))
        splits = self.splitText(data)
        st.markdown("Text Split")
        self.sum_splits(splits)
        st.markdown(self.summary)
        
    def sum_splits(self,splits):
        summarizer = pipeline("summarization", model="Falconsai/text_summarization")
        for split in splits:
            
            sum = summarizer(split.page_content, max_length=10, min_length=1, do_sample=False)
            for s in sum:
                self.summary+=" "+s["summary_text"]
        
    def splitText(self,data):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=5)
        all_splits = text_splitter.split_documents(data)
        return all_splits
        # return data.split("\n")
    
st.header("Blockchain of Health")
# file = st.file_uploader("Upload your file securely")
# if file:
#     if not os.path.exists("tempDir/"):
#         os.makedirs("tempDir/")
#     dir="tempDir/"+file.name
#     with open(dir,"wb") as f:
#         f.write(file.getbuffer())
sum = Summarize("C:/Users/erinj/Downloads/erin.pdf")
sum.callRag()