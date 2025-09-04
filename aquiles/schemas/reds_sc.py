from aquiles.models import CreateIndex
from redis.commands.search.field import TextField, VectorField, NumericField, TagField


async def RedsSch(q: CreateIndex):
    schema = (
        TextField("name_chunk"),
        NumericField("chunk_id", sortable=True),
        NumericField("chunk_size", sortable=True),
        TextField("raw_text"),
        VectorField(
            "embedding",
            "HNSW",
            {
                "TYPE": q.dtype,
                "DIM": q.embeddings_dim,
                "DISTANCE_METRIC": "COSINE",
                "INITIAL_CAP": 400,
                "M": 16,
                "EF_CONSTRUCTION": 200,
                "EF_RUNTIME": 100,
            }
        ),
        TagField("embedding_model", separator="|"),
        TagField("author", separator="|"),
        TagField("language", separator="|"),
        TagField("topics", separator="|"),
        TagField("source", separator="|"),
        NumericField("created_at"),
        TagField("extra", separator="|")
    )

    return schema