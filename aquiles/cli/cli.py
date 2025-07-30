import click
from aquiles.configs import load_aquiles_config, save_aquiles_configs
import os
import importlib.util

@click.group()
def cli():
    """A sample CLI application."""
    pass

@cli.command("hello")
@click.option("--name")
def greet(name):
    """Greets the given name."""
    click.echo(f"Hello, {name}!")

@cli.command("configs")
@click.option("--local", type=bool, default=None, help="Set whether the Redis server runs locally")
@click.option("--host", default=None, help="Redis service host")
@click.option("--port", type=int, default=None, help="Redis service port")
@click.option("--username", default=None, help="Redis username (if any)")
@click.option("--password", default=None, help="Redis password (if any)")
@click.option("--cluster-mode", type=bool, default=None, help="Enable Redis Cluster mode")
@click.option("--tls-mode", type=bool, default=None, help="Enable SSL/TLS connection")
@click.option("--ssl-cert", default=None, help="Absolute path to SSL cert")
@click.option("--ssl-key", default=None, help="Absolute path to SSL key")
@click.option("--ssl-ca", default=None, help="Absolute path to SSL CA")
def save_configs(local, host, port,
                 username, password,
                 cluster_mode, tls_mode,
                 ssl_cert, ssl_key, ssl_ca):
    try:
        configs = load_aquiles_config()

        updates = {
            "local": local,
            "host": host,
            "port": port,
            "username": username,
            "password": password,
            "cluster_mode": cluster_mode,
            "tls_mode": tls_mode,
            "ssl_cert": ssl_cert,
            "ssl_key": ssl_key,
            "ssl_ca": ssl_ca,
        }
        for key, val in updates.items():
            if val is not None:
                configs[key] = val

        save_aquiles_configs(configs)
        click.echo("✅ Configuration updated successfully.")

    except Exception as e:
        click.echo(f"❌ Error saving configuration: {e}")

@cli.command("serve")
@click.option("--host", default="0.0.0.0", help="Host where Aquiles-RAG will be executed")
@click.option("--port", type=int, default=5500, help="Port where Aquiles-RAG will be executed")
def serve(host, port):
    """Inicia el servidor FastAPI de Aquiles-RAG."""
    import uvicorn
    from aquiles.main import app
    uvicorn.run(app, host=host, port=port)

@cli.command("deploy")
@click.option("--host", default="0.0.0.0", help="Host where Aquiles-RAG will be executed")
@click.option("--port", type=int, default=5500, help="Port where Aquiles-RAG will be executed")
@click.option("--workers", type=int, default=4, help="Number of uvicorn workers when casting Aquiles-RAG")
@click.argument("config", type=click.Path(exists=True))
def deploy_command(host, port, config, workers):
    import subprocess

    module_name = os.path.splitext(os.path.basename(config))[0]
    spec = importlib.util.spec_from_file_location(module_name, config)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "run"):
        module.run()
    else:
        click.echo("The file does not have a 'run()' function")

    cmd = [
        "uvicorn",
        "aquiles.main:app",   
        "--host", str(host),
        "--port", str(port),
        "--workers", str(workers)
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    cli()