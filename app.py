from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY


embeddings = download_embeddings()

index_name = "medical-chatbot"
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding = embeddings
)


# rag chain
retriever = docsearch.as_retriever(search_type = "similarity" , search_kwargs = {'k':3})

ChatModel = AzureChatOpenAI(
    api_version = "2024-06-01",
    azure_endpoint = "https://hassan-siddiqui.openai.azure.com/",
    api_key = AZURE_OPENAI_API_KEY,
    deployment_name = "gpt-4o"
)
prompt = ChatPromptTemplate.from_messages(
    [
    ("system", system_prompt),
    ("human", "{input}"),
    ]
)

question_answer_Chain = create_stuff_documents_chain(ChatModel , prompt)
rag_chain = create_retrieval_chain(retriever , question_answer_Chain)


# route
@app.route("/")
def index():
    return render_template('chat.html')


# when ever the user click on the send button on frontend this route will be used
@app.route("/get" , methods = ["GET" , "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input" : msg})
    print("Response : " , response["answer"])
    return str(response["answer"])



if __name__ == '__main__':
    app.run(host="0.0.0.0" , port = 8080 , debug = True)