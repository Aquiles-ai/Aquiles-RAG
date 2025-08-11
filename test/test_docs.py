"""
This test or example file was created after the comment "How do you handle document chunking?" 
by user HarambeTenSei. Post: https://www.reddit.com/r/Rag/comments/1mm3qh2/aquilesrag_a_highperformance_rag_server/

This file gives you a minimal example of how Aquiles-RAG makes text chunks automatically, 
relying on other modules to extract the text. 
Remember that Aquiles-RAG is completely agnostic to the embedding model. I hope it helps :D
"""

from Artemisa.Extractor import PDFExtractor # To install this: pip install Artemisa || It's a module I created a long time ago for other tasks, but I wrote some very good tools to extract information from PDF, Docs and other documents.
from aquiles.client import AquilesRAG # We always recommend using the AsyncAquilesRAG client as it gives better performance, but as it is an example we do it with the normal client
from openai import OpenAI
from pathlib import Path

def gen_emb(text):
    client = OpenAI()
    embedding = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )

    return embedding.data[0].embedding

def index_docs(path_pdf):
    client = AquilesRAG(host="http://127.0.0.1:5500", api_key="dummy-api-key")

    print("Extracting information from the document\n")
    extractor = PDFExtractor(path_pdf)
    pages = extractor.extract_all()["pages"]
    content = [p["text"] for p in pages if p.get("text")]

    print("Sending the information to the RAG\n")
    client.send_rag(embedding_func=gen_emb, index="docs2", name_chunk=Path(path_pdf).name, raw_text=str(content))
    print("Success")

index_docs("2506.08872v1YourbrainChatGPT.pdf")