from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from aquiles.configs import load_aquiles_config
from starlette import status
from typing import Optional, List, Union, Dict
import ast
from packaging.version import Version, InvalidVersion
from importlib.metadata import version as get_installed_version, PackageNotFoundError
import requests
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.rule import Rule
from aquiles.configs import InitConfigsQdrant, InitConfigsRedis, InitConfigsPostgreSQL, AllowedUser, init_aquiles_config_v2, AQUILES_CONFIG
from getpass import getpass
from pathlib import Path
import onnxruntime as ort

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header)
):
    configs = await load_aquiles_config()
    valid_keys = [k for k in configs["allows_api_keys"] if k and k.strip()]
    
    if not valid_keys:
        return None

    if configs["allows_api_keys"]:
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API key missing",
            )
        if api_key not in configs["allows_api_keys"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key",
            )

        return api_key

def chunk_text_by_words(text: str, chunk_size: int = 600) -> list[str]:
    """
    Splits a text into chunks of up to chunk_size words.
    We will use an average of 600 words equivalent to 1024 tokens

    Args:
        text (str): Input text.
        chunk_size (int): Maximum number of words per chunk.

    Returns:
        List[str]: List of text chunks.
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
    
    return chunks


def checkout():
    pkg = "aquiles-rag"
    url = f"https://pypi.org/pypi/{pkg}/json"

    # 1) Obtiene la info desde PyPI JSON
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
    except requests.RequestException:
        # Si falla la conexión a PyPI, asumimos que no podemos comprobar
        return True, None

    data = resp.json()
    latest = data.get("info", {}).get("version")
    if not latest:
        return True, None

    # 2) Obtiene la versión local
    try:
        v_local = get_installed_version(pkg)
    except PackageNotFoundError:
        # No instalado: forzamos sugerencia de instalar la última
        return False, latest

    # 3) Compara semánticamente
    try:
        v_local_parsed  = Version(v_local)
        v_latest_parsed = Version(latest)
    except InvalidVersion:
        # Alguna versión no parsea: sugerimos actualizar
        return False, latest

    if v_local_parsed < v_latest_parsed:
        return False, latest
    else:
        return True, latest

def _escape_tag(val: str) -> str:
    return (
        str(val)
        .replace("\\", "\\\\")   
        .replace(",", "\\,")
        .replace("|", "\\|")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("-", "\\-")    
        .replace(":", "\\:")     
    )

def get_system_providers() -> List[str]:
    return ort.get_available_providers()

console = Console()


def _parse_comma_list(s: str) -> List[str]:
    return [item.strip() for item in s.split(",") if item.strip()]


def _display_summary(cfg) -> None:
    """Show a compact summary table (hiding sensitive fields)."""
    # convert to dict in a Pydantic v1/v2 compatible way
    try:
        cfg_d = cfg.dict(exclude_none=True)
    except Exception:
        cfg_d = cfg.model_dump(exclude_none=True)

    # remove or mask sensitive fields
    cfg_d.pop("password", None)
    if "allows_users" in cfg_d:
        # show only usernames for the summary
        cfg_d["allows_users"] = [u["username"] if isinstance(u, dict) else getattr(u, "username", "<user>") for u in cfg_d["allows_users"]]

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Field", style="dim", width=24)
    table.add_column("Value", overflow="fold")

    for k, v in cfg_d.items():
        table.add_row(k, str(v))

    console.print(Panel(table, title="Configuration Summary", subtitle_align="right"))


def _extract_text_from_chunk(chunk: Union[Dict, str]) -> str:
    if isinstance(chunk, dict):
        raw = chunk.get("raw_text") or chunk.get("text") or chunk.get("content") or ""
    else:
        raw = str(chunk)

    if isinstance(raw, str) and raw.strip().startswith("["):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)):
                raw = "\n".join(str(x) for x in parsed)
            else:
                raw = str(parsed)
        except Exception:
            pass

    return raw


def create_config_cli(checkout: bool = True) -> None:
    """
    Interactive Rich CLI to create an Aquiles configuration for Redis, Qdrant or PostgreSQL.

    - If checkout=True and AQUILES_CONFIG exists -> do nothing.
    - If checkout=False -> always prompt and save (overwrites).
    """
    config_path = Path(AQUILES_CONFIG)

    # respect checkout: if the file already exists, do not launch CLI
    if checkout and config_path.exists():
        console.print(
            Panel(
                "[green]:white_check_mark: Configuration already exists[/green]\n"
                f"[grey62]{config_path}[/grey62]\n\n"
                "If you want to reconfigure, call this function with [bold]checkout=False[/bold].",
                title="Aquiles-RAG",
                subtitle="No action required",
            )
        )
        return

    console.print(Rule("[bold magenta]Aquiles-RAG Configuration Wizard[/bold magenta]"))
    console.print(Panel(":gorilla:  Welcome to [bold magenta]Aquiles-RAG[/bold magenta]!\n\n"
                        "I'll ask a few questions to create your configuration file.", title="Hello", subtitle="Let's get started"))

    choice = Prompt.ask(
        "Which database would you like to configure?  (1) Redis  (2) Qdrant  (3) PostgreSQL",
        choices=["1", "2", "3"],
        default="1",
    )

    # Helper function for reranker configuration
    def _configure_reranker():
        """Configure reranker settings common to all database types"""
        rerank = Confirm.ask("Enable reranker functionality?", default=False)
        
        if not rerank:
            return {
                'rerank': False,
                'provider_re': None,
                'reranker_model': None,
                'max_concurrent_request': None,
                'reranker_preload': None
            }
        
        console.print("\n[bold yellow]Reranker Configuration:[/bold yellow]")
        
        provider_choices = {
            "1": "CPUExecutionProvider",
            "2": "CUDAExecutionProvider", 
            "3": "ROCMExecutionProvider",
            "4": "TensorrtExecutionProvider",
            "5": "DmlExecutionProvider",
            "6": "OpenVINOExecutionProvider",
            "7": "CoreMLExecutionProvider"
        }
        
        console.print("Available providers:")
        for key, provider in provider_choices.items():
            console.print(f"  {key}) {provider}")
        
        provider_choice = Prompt.ask(
            "Select execution provider",
            choices=list(provider_choices.keys()),
            default="1"
        )
        provider_re = provider_choices[provider_choice]
        
        reranker_model = Prompt.ask(
            "Reranker model name (leave empty for default)",
            default=""
        ) or None
        
        max_concurrent_str = Prompt.ask(
            "Maximum concurrent requests (leave empty for default)",
            default=""
        )
        max_concurrent_request = int(max_concurrent_str) if max_concurrent_str else None
        
        reranker_preload = Confirm.ask("Preload reranker model into memory?", default=False)
        
        return {
            'rerank': rerank,
            'provider_re': provider_re,
            'reranker_model': reranker_model,
            'max_concurrent_request': max_concurrent_request,
            'reranker_preload': reranker_preload
        }

    if choice == "1":
        console.print(":fire: [bold red]Redis selected[/bold red]\n")
        local = Confirm.ask("Is Redis running locally?", default=True)
        host = Prompt.ask("Redis host", default="localhost")
        port = int(Prompt.ask("Redis port", default="6379"))
        username = Prompt.ask("Redis username (leave empty if none)", default="")
        password = getpass("Redis password (leave empty if none): ")
        cluster_mode = Confirm.ask("Is this a Redis Cluster?", default=False)
        tls_mode = Confirm.ask("Use TLS/SSL for Redis?", default=False)
        ssl_cert = ssl_key = ssl_ca = ""
        if tls_mode:
            ssl_cert = Prompt.ask("Path to ssl_cert (absolute)", default="")
            ssl_key = Prompt.ask("Path to ssl_key (absolute)", default="")
            ssl_ca = Prompt.ask("Path to ssl_ca (optional)", default="")
        allows_api_keys = _parse_comma_list(Prompt.ask("Allowed API keys (comma separated)", default=""))

        add_admin = Confirm.ask("Create an admin user (username/password)?", default=True)
        if add_admin:
            admin_user = Prompt.ask("Admin username", default="root")
            admin_pass = getpass("Admin password: ")
            allows_users = [AllowedUser(username=admin_user, password=admin_pass)]
        else:
            allows_users = []

        reranker_config = _configure_reranker()

        cfg = InitConfigsRedis(
            local=local,
            host=host,
            port=port,
            username=username,
            password=password,
            cluster_mode=cluster_mode,
            tls_mode=tls_mode,
            ssl_cert=ssl_cert,
            ssl_key=ssl_key,
            ssl_ca=ssl_ca,
            allows_api_keys=allows_api_keys,
            allows_users=allows_users or [AllowedUser(username="root", password="root")],
            initial_cap=400,
            **reranker_config
        )

    elif choice == "2":
        console.print(":satellite: [bold cyan]Qdrant selected[/bold cyan]\n")
        local = Confirm.ask("Is Qdrant running locally?", default=True)
        host = Prompt.ask("Qdrant host", default="localhost")
        port = int(Prompt.ask("Qdrant port", default="6333"))
        prefer_grpc = Confirm.ask("Prefer gRPC instead of HTTP?", default=False)
        grpc_port = int(Prompt.ask("gRPC port (if applicable)", default="6334"))
        api_key = Prompt.ask("Qdrant API key (leave empty if none)", default="")
        auth_token_provider = Prompt.ask("Auth token provider (leave empty if none)", default="")
        allows_api_keys = _parse_comma_list(Prompt.ask("Allowed API keys (comma separated)", default=""))

        add_admin = Confirm.ask("Create an admin user (username/password)?", default=True)
        if add_admin:
            admin_user = Prompt.ask("Admin username", default="root")
            admin_pass = getpass("Admin password: ")
            allows_users = [AllowedUser(username=admin_user, password=admin_pass)]
        else:
            allows_users = []

        reranker_config = _configure_reranker()

        cfg = InitConfigsQdrant(
            local=local,
            host=host,
            port=port,
            prefer_grpc=prefer_grpc,
            grpc_port=grpc_port,
            api_key=api_key,
            auth_token_provider=auth_token_provider,
            allows_api_keys=allows_api_keys,
            allows_users=allows_users or [AllowedUser(username="root", password="root")],
            **reranker_config
        )

    else:
        console.print(":elephant:  [bold green]PostgreSQL selected[/bold green]\n")
        local = Confirm.ask("Is PostgreSQL running locally?", default=True)
        host = Prompt.ask("PostgreSQL host", default="localhost")
        port = int(Prompt.ask("PostgreSQL port", default="5432"))
        user = Prompt.ask("PostgreSQL user (leave empty to use peer/ident)", default="")
        password = getpass("PostgreSQL password (leave empty if none): ")
        database = Prompt.ask("Database name", default="postgres")
        min_size = int(Prompt.ask("Connection pool min_size", default="1"))
        max_size = int(Prompt.ask("Connection pool max_size", default="10"))
        max_queries = int(Prompt.ask("Connection pool max_queries", default="50000"))
        timeout = float(Prompt.ask("Connection timeout (seconds)", default="60"))
        allows_api_keys = _parse_comma_list(Prompt.ask("Allowed API keys (comma separated)", default=""))

        add_admin = Confirm.ask("Create an admin user (username/password)?", default=True)
        if add_admin:
            admin_user = Prompt.ask("Admin username", default="root")
            admin_pass = getpass("Admin password: ")
            allows_users = [AllowedUser(username=admin_user, password=admin_pass)]
        else:
            allows_users = []

        reranker_config = _configure_reranker()

        cfg = InitConfigsPostgreSQL(
            type_c="PostgreSQL",
            local=local,
            host=host,
            port=port,
            user=user or None,
            password=password or None,
            database=database or None,
            min_size=min_size,
            max_size=max_size,
            max_queries=max_queries,
            timeout=timeout,
            allows_api_keys=allows_api_keys,
            allows_users=allows_users or [AllowedUser(username="root", password="root")],
            **reranker_config
        )

    console.print(Rule())
    _display_summary(cfg)

    if Confirm.ask("\nSave this configuration to disk?", default=True):
        init_aquiles_config_v2(cfg, force=True)
        console.print(Panel(f":floppy_disk:  [green]Configuration saved to[/green] [bold]{AQUILES_CONFIG}[/bold]",
                            title="Done", style="green"))
    else:
        console.print(Panel("[yellow]No changes made. Configuration was not saved.[/yellow]", title="Cancelled", style="yellow"))