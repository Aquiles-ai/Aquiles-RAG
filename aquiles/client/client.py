import requests as r
from typing import Literal

class AquilesRAG:
    def __init__(self, host: str = "http://127.0.0.1:5500"):
        """ Client for the Aquiles-RAG server """
        self.base_url = host

    def create_index(self, index_name: str, 
            embeddings_dim: int = 768, 
            dtype: Literal["FLOAT32", "FLOAT64", "FLOAT16"] = "FLOAT32",
            delete_the_index_if_it_exists: bool = False):
        url = f'{self.base_url}/create/index'
        body = {"indexname" : index_name,
                "embeddings_dim": embeddings_dim,
                "dtype": dtype,
                "delete_the_index_if_it_exists": delete_the_index_if_it_exists}
        try:
            response = r.post(url=url, json=body)
            return response.text
        except Exception as e:
            print(f"Error: {e}")
            return None

    def get_configs(self):
        url = f'{self.base_url}/ui/configs'
        try:
            response = r.get(url=url)
            return response.text
        except Exception as e:
            print(f"Error: {e}")
            return None

    def edits_configs(self, local=None, host=None, port=None,
            username=None, password=None,
            cluster_mode=None, tls_mode=None,
            ssl_cert=None, ssl_key=None, ssl_ca=None):

            base_url = f'{self.base_url}/ui/configs'

            candidates = {
                "local": local,
                "host": host,
                "port": port,
                "usernanme": username,
                "password": password,
                "cluster_mode": cluster_mode,
                "tls_mode": tls_mode,
                "ssl_cert": ssl_cert,
                "ssl_key": ssl_key,
                "ssl_ca": ssl_ca,
                }

            updates = {k: v for k, v in candidates.items() if v is not None}

            if not updates:
                print("No hay ningún parámetro para actualizar.")
                return None

            response = r.post(url=base_url, json=updates)

            try:
                response.raise_for_status()
            except r.HTTPError as e:
                print(f"Error HTTP: {e} – {response.text}")
                return None
    
            return response.json()