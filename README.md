# LM Studio Streaming Chatbot

This project is a Streamlit application that processes PDF documents and answers questions based on the content using a chatbot interface. The chatbot utilizes Langchain, OpenAI, SentenceTransformers, ChromaDB, and other libraries.

## Prerequisites

- Python 3.7+
- LM Studio 

## Installation
### Create a virtual environment to manage dependencies
`python -m venv myenv`

`source myenv/bin/activate`  --On Windows use `myenv\Scripts\activate`

### Install required Libraries
`pip install streamlit langchain openai sentence-transformers chromadb PyPDF2`

### Start LM Studio Local Inference Server:
Ensure LM Studio Local Inference Server is running on `http://localhost:1234/v1`.
If not installed, refer to the LM Studio documentation for installation and setup instructions.

### Prepare the Directory Structure:
Ensure you have a directory for uploading PDFs.
mkdir uploaded_pdfs

### Download the Script:
The provided 'chatbot_app.py' contains Streamlit script.

### Run the Streamlit Application
streamlit run chatbot_app.py

## Interact with the application in the browser
Open a browser and navigate to the URL provided by Streamlit (usually http://localhost:8501).
Use the sidebar to upload a PDF document and click "Process".
Once the PDF is processed, interact with the chatbot by typing messages in the chat input field.

### By following these steps, you should be able to set up and run the Streamlit chatbot application successfully.
