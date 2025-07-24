from aquiles.client import AquilesRAG

client = AquilesRAG(api_key="dummy-api-key")

print(client.create_index("docs", embeddings_dim=1536, dtype="FLOAT32"))