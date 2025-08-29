import click
from aquiles.utils import create_config_cli
import os
import importlib.util
from aquiles.utils import checkout

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
def save_configs():
    try:
        create_config_cli(False)
    except Exception as e:
        click.echo(f"❌ Error saving configuration: {e}")

@cli.command("serve")
@click.option("--host", default="0.0.0.0", help="Host where Aquiles-RAG will be executed")
@click.option("--port", type=int, default=5500, help="Port where Aquiles-RAG will be executed")
def serve(host, port):
    """Inicia el servidor FastAPI de Aquiles-RAG."""
    try:
        import uvicorn
        from aquiles.main import app
        create_config_cli()
        uvicorn.run(app, host=host, port=port)
    finally:
        up_to_date, latest = checkout()
        if not up_to_date and latest:
            click.secho(
                f"🚀 A new version is available: aquiles-rag=={latest}\n"
                f"Update with:\n"
                f"   pip install aquiles-rag=={latest}",
                fg="yellow",
            )

@cli.command("deploy")
@click.option("--host", default="0.0.0.0", help="Host where Aquiles-RAG will be executed")
@click.option("--port", type=int, default=5500, help="Port where Aquiles-RAG will be executed")
@click.option("--workers", type=int, default=4, help="Number of uvicorn workers when casting Aquiles-RAG")
@click.argument("config", type=click.Path(exists=True))
def deploy_command(host, port, config, workers):
    up_to_date, latest = checkout()
    if not up_to_date and latest:
        click.secho(
            f"🚀 A new version is available: aquiles-rag=={latest}\n"
            f"  Update with:\n"
            f"    pip install aquiles-rag=={latest}",
            fg="yellow",
        )
    
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