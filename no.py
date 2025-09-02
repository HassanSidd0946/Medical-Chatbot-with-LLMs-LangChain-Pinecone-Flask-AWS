from langchain.document_loaders import PyPDFLoader , DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


from typing import List
from langchain.schema import Document

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

minial_docs = filter_to_minimal_docs()



def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    texts = text_splitter.split_documents(minimal_docs)
    return texts


from langchain.embeddings import HuggingFaceBgeEmbeddings

def download_embeddings():
    """
    Download and return the HuggingFace embeddings model.
    """
    model_name = "BAAI/bge-small-en-v1.5"
    embeddings = HuggingFaceBgeEmbeddings(
        model_name = model_name,
        model_kwargs = {"device" : "cuda" if torch.cuda.is_available() else "cpu"}
    )
    return embeddings

embedding = download_embeddings()

Chatmodel = AzureChatOpenAI(
    model = "gpt4o"
)
