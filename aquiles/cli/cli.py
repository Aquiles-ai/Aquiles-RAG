import click
from configs import load_aquiles_config, save_aquiles_configs

@click.group()
def cli():
    """A sample CLI application."""
    pass

@cli.command("hello")
@click.argument("--name")
def greet(name):
    """Greets the given name."""
    click.echo(f"Hello, {name}!")

@cli.command("configs")
@click.option("--local", default=True, help="Option to set whether the Redis server runs locally or not")
@click.option("--host", default="localhost", help="We configure the Redis service host")
@click.option("--port", default=6379, help="Redis service port")
def save_configs(local, host, port):
    try:
        configs = load_aquiles_config()
        configs["local"] = local
        configs["host"] = host
        configs["port"] = port
        save_aquiles_configs(configs)

    except Exception as e:
        click.echo(f"Error: {e}!")

@cli.command("serve")
def serve():
    """Inicia el servidor FastAPI de Aquiles-RAG."""
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5500)


if __name__ == "__main__":
    cli()