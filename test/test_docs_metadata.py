"""
Because the new APIs allow both sending and searching to the RAG with metadata, 
this file has been left so you can use this new feature and we will update the 
documentation with these features very soon :D
"""

from openai import OpenAI
from pathlib import Path
from aquiles.client import AquilesRAG
from Artemisa.Extractor import PDFExtractor

def gen_emb(text):
    client = OpenAI()
    embedding = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )

    return embedding.data[0].embedding

url = 'http://192.168.1.20:5500'

def new_index(index: str, embeddings_dim: int):
    client = AquilesRAG(host=url, api_key="dummy-api-key")

    print(client.create_index(index, embeddings_dim))


# testdocs

new_index("testdocs4", 1536)

path_pdf = "2502.09992v2LargeLanguageDiffusionModels.pdf"

def index_docs(path_pdf):
    client = AquilesRAG(host=url, api_key="dummy-api-key")

    print("Extracting information from the document\n")
    extractor = PDFExtractor(path_pdf)
    pages = extractor.extract_all()["pages"]
    content = [p["text"] for p in pages if p.get("text")]

    metadata = {"author": "Xiaolu Zhang",
                "language": "EN",
                "topics": list({"Diffusion", "LLM", "LLaDA"}),
                "source": "arxiv"}

    print("Sending the information to the RAG\n")
    s = client.send_rag(embedding_func=gen_emb, index="testdocs4", name_chunk=Path(path_pdf).name, 
    raw_text=str(content), embedding_model="text-embedding-3-small", metadata=metadata)
    print(f"Success, Response: {s}")

index_docs(path_pdf)

embe = gen_emb("The core of LLaDA")

def query(embedding):
    client = AquilesRAG(host=url, api_key="dummy-api-key")

    metadata = {"author": "Xiaolu Zhang",
                "language": "EN",
                "source": "arxiv"}

    response = client.query("testdocs4", embedding=embedding, embedding_model="text-embedding-3-small", metadata=metadata)

    print(response)

query(embedding=embe)
