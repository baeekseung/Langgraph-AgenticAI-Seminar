from fastmcp import FastMCP

# FastMCP 서버 초기화 및 구성
mcp = FastMCP(
    "AddTwoNumbers",  # MCP 서버 이름
    instructions="두 수를 더하는 어시스턴트입니다. 주어진 두 수에 대한 결과를 반환할 수 있습니다.",
)


@mcp.tool()
async def add_two_numbers(a: int, b: int) -> int:
    """두 수를 더합니다. 주어진 두 수에 대한 결과를 반환합니다.
    Args:
        a: 첫 번째 수
        b: 두 번째 수

    Returns:
        두 수의 합
    """
    return a + b


if __name__ == "__main__":
    # stdio 전송 방식으로 MCP 서버를 시작합니다
    # stdio 전송은 표준 입출력 스트림을 통해 클라이언트와 통신하며,
    # 로컬 개발 및 테스트에 적합합니다
    mcp.run(transport="stdio")
