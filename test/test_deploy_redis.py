"""
This is an example of a file that will be used to deploy Aquiles-RAG to providers 
like Render using Redis as the RAG, you have to create a requirements.txt with "aquiles-rag" as 
the only module to install, and in the command to launch the service 
you have to use "quiles-rag deploy --host "0.0.0.0" --port 5500 --workers 4 your_config_file.py"
"""
import os
from dotenv import load_dotenv
from pathlib import Path
from aquiles.deploy_config import DeployConfigRd, gen_configs_file
from aquiles.configs import AllowedUser

# You must set all configuration options with the 'DeployConfigRd' class

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

REDIS_HOST = os.getenv('REDIS_HOST', 'redis-dummy.com')
REDIS_PORT = int(os.getenv('REDIS_PORT', 123))
REDIS_USER = os.getenv('REDIS_USERNAME', 'default')
REDIS_PASSWORD = os.getenv('REDIS_PASS', 'dummy-password')

apikeys = ["dummy-api-key", "secure-api-key"]

users = [AllowedUser(username="root", password="root"),
        AllowedUser(username="supersu", password="supersu")]

dp_cfg = DeployConfigRd(
    local=False, host=REDIS_HOST,
    port=REDIS_PORT, username=REDIS_USER, password=REDIS_PASSWORD, cluster_mode=False,
    tls_mode=False, ssl_cert="", ssl_key="", ssl_ca="",
    allows_api_keys=apikeys,
    allows_users=users,
    initial_cap=200,
    rerank=False,
    provider_re=None,
    reranker_model=None,
    max_concurrent_request=None,
    reranker_preload=None,
    ALGORITHM="HS256"
)

# Make sure that when generating the config files you encapsulate it in a 'run' function

def run():
    print("Generating the configs file")
    gen_configs_file(dp_cfg, force=True)

