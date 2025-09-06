"""
This is an example of a file that will be used to deploy Aquiles-RAG to providers 
like Render using Qdrant as the RAG, you have to create a requirements.txt with "aquiles-rag" as 
the only module to install, and in the command to launch the service 
you have to use "quiles-rag deploy --host "0.0.0.0" --port 5500 --workers 4 your_config_file.py"
"""

import os
from dotenv import load_dotenv
from pathlib import Path
from aquiles.deploy_config import DeployConfigQdrant, gen_configs_file
from aquiles.configs import AllowedUser

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

HOST = os.getenv('HOST', 'fe2.gcp.cloud.qdrant.io')
PORT = int(os.getenv('PORT', 6333))
API_KEY_QDRANT = os.getenv('API_KEY_QDRANT', 'dummy-api-key')

# If you are going to use gRPC make sure you have the gRPC port and set the gRPC options

GPRC_PORT = int(os.getenv('GPRC_PORT', 6334))

# GRPC_OPT = os.getenv('GRPC_OPT')

API_KEYS = ["dummy-api-key", "idk-api-key"]

users = [AllowedUser(username="root", password="root"), 
            AllowedUser(username="supersu", password="supersu")]

dp_cfg = DeployConfigQdrant(local=False,
        host=HOST,
        port=PORT,
        prefer_grpc=False,
        grpc_port=GPRC_PORT,
        api_key=API_KEY_QDRANT,
        allows_api_keys=API_KEYS,
        allows_users=users,
        rerank=False,
        provider_re=None,
        reranker_model=None,
        max_concurrent_request=None,
        reranker_preload=None,
        ALGORITHM="HS256")

def run():
    print("Generating the configs file")
    gen_configs_file(dp_cfg, force=True)