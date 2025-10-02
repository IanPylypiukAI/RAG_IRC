"""
Module for processing the PDF version of the US Internal Revenue Code.

Before running in Google Colab:
- Switch runtime to GPU (x60 faster than CPU).
- Add the required secrets via Colab:
  - HUGGINGFACE_TAX_APP_SECRET
  - QDRANT_CLIENT_API_KEY
  - QDRANT_CLIENT_URL
"""

!pip install requests pymupdf
!pip install llama-index llama-index-embeddings-huggingface llama-index-readers-file
!pip install qdrant-client llama-index-vector-stores-qdrant

import os
import shutil
import zipfile
import requests
import logging
import torch

from huggingface_hub import login
from google.colab import userdata

from llama_index.readers.file import PyMuPDFReader
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore


# ====== LOGGING SETUP ====== #

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)


# ====== FUNCTIONS ====== #

def get_device():
    if torch.cuda.is_available():
        logging.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    elif torch.backends.mps.is_available():
        logging.info("Using Apple MPS")
        return "mps"
    else:
        logging.info("Using CPU")
        return "cpu"

def download_zip(url: str, zip_path: str, timeout: int = 360):
    """Streaming downloading"""
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        logging.info(f'File downloaded successfully: {zip_path}')
    except requests.exceptions.HTTPError as e:
        logging.error(f'HTTP error: {e}')
        raise
    except requests.exceptions.ConnectionError as e:
        logging.error(f'Connection error: {e}')
        raise
    except requests.exceptions.Timeout as e:
        logging.error(f'Timeout error: {e}')
        raise
    except Exception as e:
        logging.error(f'Other error: {e}')
        raise

def unpack_zip(zip_path: str, extract_to: str):
    # Unpack ZIP and return file list
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
        file_list = z.namelist()

    os.remove(zip_path)
    return file_list

def load_pdf(pdf_path: str):
    """Reading PDF with PyMuPDFReader and filtering blank pages"""
    loader = PyMuPDFReader()
    try:
        documents = loader.load(file_path=pdf_path)
    except Exception as e:
        logging.error(f'Error reading PDF: {e}')
        documents = []

    cleaned_documents = []
    for i, doc in enumerate(documents):
        text = getattr(doc, 'text', '')
        if text.strip():
            metadata = doc.metadata or {}
            metadata.update({
                'page': i,
                'source_file': os.path.basename(pdf_path)
            })
            cleaned_documents.append(Document(text=text, metadata=metadata))

    if not cleaned_documents:
        raise Exception('No documents with text found')
    logging.info(f'Loaded {len(cleaned_documents)} documents from PDF')
    return cleaned_documents

def create_qdrant_index(nodes, collection_name: str, embedding_model):
    """Create collection in Qdrant and push embeddings"""
    QDRANT_CLIENT_URL = userdata.get('QDRANT_CLIENT_URL')
    QDRANT_CLIENT_API_KEY = userdata.get('QDRANT_CLIENT_API_KEY')
    qdrant_client = QdrantClient(url=QDRANT_CLIENT_URL, api_key=QDRANT_CLIENT_API_KEY)

    # Check collection
    existing = [c.name for c in qdrant_client.get_collections().collections]
    if collection_name not in existing:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vector_size=embedding_model.embedding_dim,
            distance='Cosine'
        )
        logging.info(f'Created new Qdrant collection: {collection_name}')

    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Save embeddings to Qdrant
    logging.info('Saving embeddings to Qdrant...')
    VectorStoreIndex(nodes, embed_model=embedding_model, storage_context=storage_context)
    logging.info('Index created and stored in Qdrant successfully.')


# ====== MAIN SCRIPT ====== #

if __name__ == '__main__':
    # Parameters
    URL = 'https://uscode.house.gov/download/releasepoints/us/pl/119/36/pdf_usc26@119-36.zip'
    FOLDER_TO_UNZIP = 'folder_to_unzip'
    ZIP_TEMP_PATH = 'temp.zip'
    COLLECTION_NAME = 'internal-revenue-code'

    # Download and unpack ZIP
    download_zip(URL, ZIP_TEMP_PATH)
    files = unpack_zip(ZIP_TEMP_PATH, FOLDER_TO_UNZIP)

    # Take the first PDF
    pdf_files = [f for f in files if f.lower().endswith('.pdf')]
    if not pdf_files:
        raise Exception('No PDF files found in ZIP')
    pdf_path = os.path.join(FOLDER_TO_UNZIP, pdf_files[0])

    # Read PDF
    documents = load_pdf(pdf_path)

    # Cut into chunks
    parser = SimpleNodeParser.from_defaults(chunk_size=1000, chunk_overlap=100)
    nodes = parser.get_nodes_from_documents(documents)
    logging.info(f'Created {len(nodes)} nodes from documents')

    # Login to HuggingFace
    HUGGINGFACE_TAX_APP_SECRET = userdata.get('HUGGINGFACE_TAX_APP_SECRET')
    login(token=HUGGINGFACE_TAX_APP_SECRET)

    device = get_device()
    embedding_model = HuggingFaceEmbedding(
        model_name='BAAI/bge-m3',
        embed_batch_size=8,
        device=device,
    )

    # Push to Qdrant
    create_qdrant_index(nodes, COLLECTION_NAME, embedding_model)
