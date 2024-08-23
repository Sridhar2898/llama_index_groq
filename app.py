import os
import tempfile
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, ServiceContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq

import warnings
warnings.filterwarnings('ignore')

# Set the title of the Streamlit app
st.sidebar.title('RAG App')

# Fetch the Groq API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY is None:
    st.error("Groq API key is missing. Please set the GROQ_API_KEY environment variable.")
else:
    llm = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

VECTOR_DB_DIR = "./newdata"

def create_vector_index(documents):
    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = text_splitter.get_nodes_from_documents(documents, show_progress=True)
    vector_index = VectorStoreIndex.from_documents(documents=documents, service_context=service_context, node_parser=nodes)
    vector_index.storage_context.persist(VECTOR_DB_DIR)
    return vector_index

def load_vector_index():
    if os.path.exists(VECTOR_DB_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=VECTOR_DB_DIR)
        return load_index_from_storage(storage_context, service_context=service_context)
    return None

input_pdfs = "Tata Code Of Conduct.pdf"
reader = SimpleDirectoryReader(input_files=[input_pdfs])
documents = reader.load_data()
vector_index = load_vector_index()
if vector_index is None:
    vector_index = create_vector_index(documents)

# uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
# if uploaded_file is not None:
#     # Save uploaded PDF to a temporary file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         tmp.write(uploaded_file.getvalue())
#         temp_path = tmp.name

#     # Load the document and create or update the vector index
#     reader = SimpleDirectoryReader(input_files=[temp_path])
#     documents = reader.load_data()

#     if vector_index is None:
#         with st.spinner('Creating the vector index...'):
#             vector_index = create_vector_index(documents)
#             st.sidebar.write("Vector index created successfully.")
#     else:
#         with st.spinner('Updating the vector index...'):
#             # vector_index.add_documents(documents=documents)
#             vector_index = create_vector_index(documents)
#             vector_index.storage_context.persist(VECTOR_DB_DIR)
#             st.sidebar.write("Vector index updated successfully.")

#     os.unlink(temp_path)  # Remove the temporary file

# Query the vector index
if vector_index is not None:
    query = st.text_input("Enter your query")
    
    if st.button('Submit Query'):
        query_engine = vector_index.as_query_engine(service_context=service_context)
        with st.spinner('Generating the answer...'):
            resp = query_engine.query(query)
            st.write(resp.response)
else:
    st.write("No vector index found. Please upload a PDF file to create the vector database.")
