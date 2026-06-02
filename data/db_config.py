"""
数据库凭证 + SSH 隧道工厂。
凭证从 backtest/.env 读，不入库；缺失则 KeyError 立刻暴露。
"""
import os
from pathlib import Path

from sshtunnel import SSHTunnelForwarder

# 加载 backtest/.env 到 os.environ（手写解析，无外部依赖）
ENV_PATH = Path(__file__).parent.parent / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

SSH_HOST = os.environ["SSH_HOST"]
SSH_PORT = int(os.environ["SSH_PORT"])
SSH_USER = os.environ["SSH_USER"]
SSH_PASSWORD = os.environ["SSH_PASSWORD"]
DB_HOST = os.environ["DB_HOST"]
DB_PORT = int(os.environ["DB_PORT"])
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_NAME = os.environ.get("DB_NAME", "fusion_prod")

# JYDB（聚源，内网 SQL Server，直连不走隧道）
JYDB_HOST = os.environ["JYDB_HOST"]
JYDB_USER = os.environ["JYDB_USER"]
JYDB_PASSWORD = os.environ["JYDB_PASSWORD"]
JYDB_NAME = os.environ.get("JYDB_NAME", "JYDB")


def make_tunnel() -> SSHTunnelForwarder:
    """统一的 SSH 隧道工厂，所有 fetch 脚本共用"""
    return SSHTunnelForwarder(
        (SSH_HOST, SSH_PORT),
        ssh_username=SSH_USER,
        ssh_password=SSH_PASSWORD,
        remote_bind_address=(DB_HOST, DB_PORT),
    )


def make_jydb_conn():
    """JYDB 直连（内网 SQL Server），返回 pymssql 连接。ST 等基本面数据用。"""
    import pymssql
    return pymssql.connect(
        JYDB_HOST, JYDB_USER, JYDB_PASSWORD, JYDB_NAME,
        timeout=60, login_timeout=15,
    )
