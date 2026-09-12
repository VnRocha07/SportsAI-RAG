from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

PASTA_BASE = Path(__file__).resolve().parent / "base"


def criar_db():
    documentos = carregar_documentos()
    # print(documentos)
    chunks = dividir_chunks(documentos)
    vetorizar_chunks(chunks)


def carregar_documentos():
    documentos = []

    for arquivo in PASTA_BASE.glob("*.pdf"):
        leitor = PdfReader(arquivo)

        for numero_pagina, pagina in enumerate(leitor.pages):
            documentos.append(
                Document(
                    page_content=pagina.extract_text() or "",
                    metadata={
                        "source": str(arquivo),
                        "file_name": arquivo.name,
                        "page": numero_pagina,
                    }
                )
            )
    return documentos


def dividir_chunks(documentos):
    separador_documentos = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )
    chunks = separador_documentos.split_documents(documentos)
    # print(len(chunks))
    return chunks


def vetorizar_chunks(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    db = Chroma.from_documents(chunks, embeddings, persist_directory="db")


criar_db()
