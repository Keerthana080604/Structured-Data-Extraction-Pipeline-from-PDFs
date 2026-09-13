from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from dotenv import load_dotenv

import os
import tempfile
import uuid
import pandas as pd
import re


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Please create a .env file in the project root."
    )

GEMINI_API_KEY = GEMINI_API_KEY.strip().strip('"').strip("'")


# --------------------------------------------------
# Clean filename
# --------------------------------------------------

def clean_filename(filename):
    """
    Remove (number) from filename.
    Example:
        research paper (1).pdf
        -> research paper.pdf
    """
    return re.sub(r'\s\(\d+\)', '', filename)


# --------------------------------------------------
# Load PDF
# --------------------------------------------------

def get_pdf_text(uploaded_file):

    temp_file = None

    try:
        input_file = uploaded_file.read()

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        )

        temp_file.write(input_file)
        temp_file.close()

        loader = PyPDFLoader(temp_file.name)

        documents = loader.load()

        return documents

    finally:

        if (
            temp_file is not None
            and os.path.exists(temp_file.name)
        ):
            os.unlink(temp_file.name)


# --------------------------------------------------
# Split documents
# --------------------------------------------------

def split_document(
    documents,
    chunk_size=1000,
    chunk_overlap=200
):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            " "
        ]
    )

    return text_splitter.split_documents(documents)


# --------------------------------------------------
# Gemini Embeddings
# --------------------------------------------------

def get_embedding_function():

    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is empty."
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    return embeddings


# --------------------------------------------------
# Create Chroma Vector Store
# --------------------------------------------------

def create_vectorstore(
    chunks,
    embedding_function,
    file_name,
    vector_store_path="db"
):

    # Create unique IDs for chunks

    ids = [
        str(
            uuid.uuid5(
                uuid.NAMESPACE_DNS,
                doc.page_content
            )
        )
        for doc in chunks
    ]

    # Remove duplicate chunks

    unique_ids = set()

    unique_chunks = []

    unique_chunk_ids = []

    for chunk, chunk_id in zip(
        chunks,
        ids
    ):

        if chunk_id not in unique_ids:

            unique_ids.add(chunk_id)

            unique_chunks.append(chunk)

            unique_chunk_ids.append(chunk_id)

    # Create Chroma vector store

    vectorstore = Chroma.from_documents(
        documents=unique_chunks,
        collection_name=clean_filename(file_name),
        embedding=embedding_function,
        ids=unique_chunk_ids,
        persist_directory=vector_store_path
    )

    return vectorstore


# --------------------------------------------------
# Create Vector Store from PDF
# --------------------------------------------------

def create_vectorstore_from_texts(
    documents,
    file_name
):

    # Split PDF into chunks

    docs = split_document(
        documents,
        chunk_size=1000,
        chunk_overlap=200
    )

    # Create Gemini embedding function

    embedding_function = get_embedding_function()

    # Create vector store

    vectorstore = create_vectorstore(
        docs,
        embedding_function,
        file_name
    )

    return vectorstore


# --------------------------------------------------
# Structured Output Models
# --------------------------------------------------

class AnswerWithSources(BaseModel):

    answer: str = Field(
        description="Answer to the question"
    )

    sources: str = Field(
        description=(
            "Full direct text chunk from the "
            "context used to answer the question"
        )
    )

    reasoning: str = Field(
        description=(
            "Explain the reasoning based on "
            "the sources"
        )
    )


class ExtractedInfoWithSources(BaseModel):

    paper_title: AnswerWithSources

    paper_summary: AnswerWithSources

    paper_authors: AnswerWithSources

    paper_publication_date: AnswerWithSources


# --------------------------------------------------
# Format retrieved documents
# --------------------------------------------------

def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# --------------------------------------------------
# Query RAG
# --------------------------------------------------

def query_document(
    vectorstore,
    query
):

    # Gemini LLM

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0
    )

    # Retriever

    retriever = vectorstore.as_retriever(
        search_type="similarity"
    )

    # Prompt

    prompt_template = ChatPromptTemplate.from_template(
        """
        You are an assistant for question-answering tasks.

        Use ONLY the following retrieved context
        to answer the question.

        If the answer is not present in the context,
        say that you don't know.

        DO NOT MAKE UP ANYTHING.

        Context:
        {context}

        Question:
        {question}
        """
    )

    # RAG chain

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt_template
        | llm.with_structured_output(
            ExtractedInfoWithSources
        )
    )

    # Get structured response

    structured_response = rag_chain.invoke(query)

    # Convert Pydantic object to dictionary

    response_dict = structured_response.model_dump()

    # Create rows

    answer_row = []

    source_row = []

    reasoning_row = []

    for field in response_dict:

        answer_row.append(
            response_dict[field]["answer"]
        )

        source_row.append(
            response_dict[field]["sources"]
        )

        reasoning_row.append(
            response_dict[field]["reasoning"]
        )

    # Create DataFrame

    structured_response_df = pd.DataFrame(
        [
            answer_row,
            source_row,
            reasoning_row
        ],
        columns=response_dict.keys(),
        index=[
            "Answer",
            "Sources",
            "Reasoning"
        ]
    )

    return structured_response_df.T