import fitz  # PyMuPDF
import streamlit as st
from transformers import pipeline
import hashlib

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Initialize the NER model
pipe = pipeline("token-classification", model="blaze999/clinical-ner", aggregation_strategy='simple')

# Function to run NER model
def ner_model(text):
    result = pipe(text)
    return result

# Function to generate a consistent light color for each entity type
def generate_color_for_entity(entity_type):
    # Hash the entity type to get a consistent value
    hash_value = int(hashlib.md5(entity_type.encode()).hexdigest(), 16)
    # Generate a light color
    color_value = (hash_value % 0xFFFFFF) + 0x808080
    color = "#{:06x}".format(color_value & 0xFFFFFF)
    return color

# Function to highlight entities in the text
def highlight_text(text, ner_output):
    highlighted_text = ""
    last_idx = 0

    # Sort entities by their start index
    ner_output = sorted(ner_output, key=lambda x: x['start'])

    # Assign colors to each entity type
    entity_colors = {}
    for entity in ner_output:
        entity_type = entity['entity_group']
        if entity_type not in entity_colors:
            entity_colors[entity_type] = generate_color_for_entity(entity_type)

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

# Streamlit app
st.title("NER Model Output from PDF")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from the uploaded PDF
    pdf_text = extract_text_from_pdf(uploaded_file)

    st.write("Extracted Text:")
    st.write(pdf_text)

    # Run the NER model on the extracted text
    ner_output = ner_model(pdf_text)

    highlighted_text = highlight_text(pdf_text, ner_output)

    st.write("Text with Highlighted Entities:")
    st.markdown(highlighted_text, unsafe_allow_html=True)
