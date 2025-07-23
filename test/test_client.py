from aquiles.client import AquilesRAG

client = AquilesRAG(api_key="dummy-api-key")

print(client.create_index("docs2", embeddings_dim=1536, dtype="FLOAT32"))