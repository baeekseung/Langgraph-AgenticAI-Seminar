from fastmcp import FastMCP
from typing import Optional
import pytz
from datetime import datetime

# FastMCP 서버 초기화 및 구성
mcp = FastMCP(
    "MultipleTwoNumbers",  # MCP 서버 이름
    instructions="두 수를 곱하는 어시스턴트입니다. 주어진 두 수에 대한 결과를 반환할 수 있습니다.",
)


@mcp.tool()
async def multiple(a: int, b: int) -> int:
    """두 수를 곱합니다. 주어진 두 수에 대한 결과를 반환합니다.
    Args:
        a: 첫 번째 수
        b: 두 번째 수

    Returns:
        두 수의 곱
    """
    return a * b


if __name__ == "__main__":
    # 서버가 시작됨을 알리는 메시지를 출력합니다
    print("MCP MultipleTwoNumbers 서버가 실행 중입니다...")

    # streamable-http 전송 방식으로 서버를 시작합니다 (포트 8002)
    mcp.run(transport="streamable-http", port=8002)
