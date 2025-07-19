from redis.commands.search.query import Query
from redis.commands.search.field import TextField, TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.json.path import Path
import numpy as np
import redis

class baseRAG:
    def __init__(self):
        pass

    def set_embeddings(self):
        raise NotImplementedError("Dummy error")