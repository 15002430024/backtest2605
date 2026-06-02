"""
纯连通性测试：SSH 隧道 → SYYL 数据库 → 查 10 行数据
不做任何框架、缓存、rename，只确认链路通 + 肉眼看数据
凭证从 backtest/.env 读（见 db_config.py）
"""
import pymysql

from db_config import DB_NAME, DB_PASSWORD, DB_USER, make_tunnel

with make_tunnel() as tunnel:
    conn = pymysql.connect(
        host='127.0.0.1',
        port=tunnel.local_bind_port,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
    )
    cursor = conn.cursor()

    # === 测试1：ashareeodprices 查 600519.SH（茅台）最近 10 行 ===
    print("=" * 80)
    print("测试1: ashareeodprices — 600519.SH 最近 10 行")
    print("=" * 80)
    cursor.execute(
        "SELECT * FROM ashareeodprices "
        "WHERE s_info_windcode = '600519.SH' "
        "ORDER BY trade_dt DESC LIMIT 10"
    )
    columns = [desc[0] for desc in cursor.description]
    print(f"\n列名 ({len(columns)} 列): {columns}\n")
    for row in cursor.fetchall():
        print(dict(zip(columns, row)))

    # === 测试2：asharecalendar 最近 10 条 ===
    print("\n" + "=" * 80)
    print("测试2: asharecalendar — 最近 10 条")
    print("=" * 80)
    cursor.execute(
        "SELECT * FROM asharecalendar "
        "ORDER BY trade_days DESC LIMIT 10"
    )
    columns2 = [desc[0] for desc in cursor.description]
    print(f"\n列名 ({len(columns2)} 列): {columns2}\n")
    for row in cursor.fetchall():
        print(dict(zip(columns2, row)))

    conn.close()
    print("\n连接已关闭，测试完成。")
