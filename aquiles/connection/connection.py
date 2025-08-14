import redis.asyncio as redis
from redis.asyncio.cluster import RedisCluster
from aquiles.configs import load_aquiles_config
from pathlib import Path

async def get_connection():
    configs    = await load_aquiles_config()
    local      = configs.get("local", True)
    host       = configs.get("host", "localhost")
    port       = configs.get("port", 6379)
    username   = configs.get("username", "")
    password   = configs.get("password", "")
    cluster    = configs.get("cluster_mode", False)
    tls_mode   = configs.get("tls_mode", False)

    # Process SSL routes securely
    raw_cert   = configs.get("ssl_certfile", "")
    ssl_cert   = str(Path(raw_cert)) if raw_cert else None
    raw_key    = configs.get("ssl_keyfile", "")
    ssl_key    = str(Path(raw_key)) if raw_key else None
    raw_ca     = configs.get("ssl_ca_certs", "")
    ssl_ca     = str(Path(raw_ca)) if raw_ca else None

    # 1) Local cluster
    if local and cluster:
        rc = RedisCluster(host=host, port=port, decode_responses=True, max_connections=1000000)
        rc.get_nodes()
        await rc.initialize()
        return rc

    # 2) Redis standalone local
    if local:
        return redis.Redis(host=host, port=port, decode_responses=True, max_connections=1000000)

    # 3) Remote with TLS/SSL
    if tls_mode:
        ssl_opts = {}
        if ssl_cert: ssl_opts["ssl_certfile"] = ssl_cert
        if ssl_key:  ssl_opts["ssl_keyfile"]  = ssl_key
        if ssl_ca:   ssl_opts["ssl_ca_certs"] = ssl_ca

        return redis.Redis(
            host=host,
            port=port,
            username=username or None,
            password=password or None,
            ssl=True,
            decode_responses=True,
            max_connections=1000000,
            **ssl_opts
        )

    # 4) Remote without TLS
    return redis.Redis(
        host=host,
        port=port,
        username=username or None,
        password=password or None,
        decode_responses=True,
        max_connections=1000000
    )
