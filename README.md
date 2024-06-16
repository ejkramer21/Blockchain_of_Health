## Blockchain_of_Health
Utilizing Blockchain and 0G System storage nodes to provide secure upload of health documents. Documents are run through an ML model to make more digestible in laymen's terms to help people take control of their health.

## The Inspiration
Health records are challenging to understand and often include a lot of foreign jargon. When it comes to your healthcare, having transparency on what medications and procedures you are being prescribed without the compromise of security is key. We solve this problem by leveraging natural language modeling techniques that provide users with greater insight into their medical documents. Specifically by employing a named entity recognition model to identify keywords within a patient's help report and an informed summarization model that concisely explains your care pathway.

### Please watch our [Demo](https://vimeo.com/959368305?share=copy) ###

### Technologies ###
Python, NodeJS, Streamlit, HuggingFace, LangChain, RAG Pipelines, Web3 technologies

### To Run ###
cd model_runners
streamlit run frontend.py

#### Important Note ####
frontend.py contains a call to a NodeJS file called sdk.js. This file contains private information and is not pushed to this repository. To duplicate our results, you must develop and attach your own NodeJS SDK file to connect the file uploading to a Web3 storage database.
