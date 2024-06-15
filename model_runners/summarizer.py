from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline
import streamlit as st

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
        loader = self.loaders[file_type]
        load = loader(str(self.file))
        data = load.load()
        return data
        
    def callRag(self):
        data = self.getLoader()
        splits = self.splitText(data)
        self.sum_splits(splits)
        st.markdown(self.summary)
        
    def sum_splits(self,splits):
        summarizer = pipeline("summarization", model="Falconsai/text_summarization")
        for split in splits:
            
            sum = summarizer(split.page_content, max_length=10, min_length=1, do_sample=False)
            for s in sum:
                self.summary+=" "+s["summary_text"]
        
    def splitText(self,data):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=5)
        all_splits = text_splitter.split_documents(data)
        return all_splits