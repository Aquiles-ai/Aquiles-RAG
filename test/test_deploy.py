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