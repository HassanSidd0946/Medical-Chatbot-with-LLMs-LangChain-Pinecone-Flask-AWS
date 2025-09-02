from langchain.document_loaders import PyPDFLoader , DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document



# Extract Data from PDF Files
def load_pdf_files(Data):
    loader = DirectoryLoader(
        Data,
        glob = "*.pdf",
        loader_cls = PyPDFLoader
    )

    documents = loader.load()
    return documents


def filter_to_minimal_docs(Docs : List[Document]) -> list[Document]:
    """
    Given a list of Document objects, return a new list of Document objects containing only 'source' in metedata and the original page_content.
    """
    minimal_docs : List[Document] = []
    for doc in Docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":src}
            )
        )
    return minimal_docs



# Split the documents into smaller chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk


# Download the Embeddings from HuggingFace
from langchain_community.embeddings import HuggingFaceEmbeddings
def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = HuggingFaceEmbeddings(model_name=model_name)
    return model

embedding = download_embeddings()
print("Embeddings model loaded successfully!")