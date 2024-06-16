import streamlit as st
from summarizer import Summarize
# from NER import NER

import fitz  # PyMuPDF+
from transformers import pipeline
import hashlib
import json

import subprocess

def call_node_function():
    try:
        result = subprocess.run(
            ['node', 'sdk.js'],  # Ensure this path is correct
            capture_output=True,
            text=True
        )
        
        # Print detailed output for debugging
        print(f"Arguments: {result.args}")
        print(f"Return Code: {result.returncode}")
        print(f"Standard Output:\n{result.stdout}")
        print(f"Standard Error:\n{result.stderr}")

        # Raise an exception if the process exited with a non-zero status
        if result.returncode != 0:
            raise Exception(f"Node.js script exited with error code {result.returncode}")
        
        # Process the output
        output_lines = result.stdout.strip().split('\n')
        # # The last line should be the JSON string of the transaction
        # transaction_output = output_lines[-1]
        
        # # Convert the JSON string to a Python dictionary
        # transaction_data = json.loads(transaction_output)
        
        return output_lines
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


class NER:
    uploaded_file=""
    pipe=None
    
    def __init__(self,file):
        self.uploaded_file=file

    # Function to extract text from PDF
    def extract_text_from_pdf(self, pdf_file):
        doc = fitz.open(pdf_file)
    
        text = ""
        # Iterate over each page and extract text
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    
    def run_NER(self):
        # Initialize the NER model
        self.pipe = pipeline("token-classification", model="blaze999/clinical-ner", aggregation_strategy='simple')
        # Streamlit app
        if self.uploaded_file is not None:
            # Extract text from the uploaded PDF
            pdf_text = self.extract_text_from_pdf(self.uploaded_file)

            # Run the NER model on the extracted text
            ner_output = self.ner_model(pdf_text)

            highlighted_text = self.highlight_text(pdf_text, ner_output)

            st.write("Text with Highlighted Entities:")
            st.markdown(highlighted_text, unsafe_allow_html=True)

    # Function to run NER model
    def ner_model(self,text):
        result = self.pipe(text)
        return result

    # Function to generate a consistent light color for each entity type
    def generate_color_for_entity(self,entity_type):
        # Hash the entity type to get a consistent value
        hash_value = int(hashlib.md5(entity_type.encode()).hexdigest(), 16)
        # Generate a light color
        color_value = (hash_value % 0xFFFFFF) + 0x808080
        color = "#{:06x}".format(color_value & 0xFFFFFF)
        return color

    # Function to highlight entities in the text
    def highlight_text(self,text, ner_output):
        highlighted_text = ""
        last_idx = 0

        # Sort entities by their start index
        ner_output = sorted(ner_output, key=lambda x: x['start'])

        # Assign colors to each entity type
        entity_colors = {}
        for entity in ner_output:
            entity_type = entity['entity_group']
            if entity_type not in entity_colors:
                entity_colors[entity_type] = self.generate_color_for_entity(entity_type)

        css = "<style>"
        for entity_type, color in entity_colors.items():
            css += f"""
            .{entity_type} {{
                background-color: {color};
                padding: 2px 6px;
                border-radius: 4px;
                display: inline-block;
                margin: 2px;
            }}
            .{entity_type} .label {{
                font-size: 10px;
                color: #fff;
                background-color: rgba(0, 0, 0, 0.6);
                border-radius: 3px;
                padding: 0 3px;
                margin-left: 5px;
            }}
            """
        css += "</style>"

        for entity in ner_output:
            start = entity['start']
            end = entity['end']
            entity_text = entity['word']
            entity_type = entity['entity_group']

            # Add text before the entity
            highlighted_text += text[last_idx:start]
            # Highlight the entity
            highlighted_text += f'<span class="{entity_type}">{entity_text}<span class="label">{entity_type}</span></span>'
            last_idx = end

        # Add the remaining text
        highlighted_text += text[last_idx:]

        return css + highlighted_text


st.markdown("<h1 style='text-align: center;'>Blockchain of Health</h1>",unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
col1, spacer, col2 = st.columns([3, 1, 3])
if uploaded_file is not None:
    print(call_node_function())
    sum = Summarize(uploaded_file.name)
    ner = NER(uploaded_file.name)
    with col2:  
        st.markdown("<h3> Summary of the document</h3>",unsafe_allow_html=True)
        
    with col1:
        st.markdown("<h3>NER Model Output from PDF</h3>",unsafe_allow_html=True)
        
    with col1:
        ner.run_NER()
    with col2:
        st.markdown(sum.callRag())
    