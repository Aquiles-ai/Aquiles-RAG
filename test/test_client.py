from aquiles.client import AquilesRAG

client = AquilesRAG()

print(client.create_index("docs", embeddings_dim=1536, dtype="FLOAT32"))