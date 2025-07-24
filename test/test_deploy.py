"""
This is an example of a file that will be used to deploy Aquiles-RAG in 
providers like Render, you have to create a requirements.txt with "aquiles-rag" as 
the only module to install, and in the command to launch the service 
you have to use "quiles-rag deploy --host "0.0.0.0" --port 5500 your_config_file.py"
"""
from aquiles.deploy_config import DeployConfig, gen_configs_file
from aquiles.configs import AllowedUser

# You must set all configuration options with the 'DeployConfig' class

dp_cfg = DeployConfig(local=True, host="",port=900,usernanme="",
    password="", cluster_mode=False, tls_mode=False, ssl_cert="",
    ssl_key="", ssl_ca="", allows_api_keys=[""], allows_users=[AllowedUser(username="root", password="root")],
    ALGORITHM="HS256"
)

# Make sure that when generating the config files you encapsulate it in a 'run' function

def run():
    print("Generating the configs file")
    gen_configs_file(dp_cfg)