from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from typing import Any


load_dotenv(override=True)

def create_retriever() -> Any:
    embedding_function = OpenAIEmbeddings(model='text-embedding-3-large')
    vector_store = Chroma(
        embedding_function=embedding_function,
        collection_name = 'chroma_collection',
        persist_directory = './chroma_collection'
    )
    retriever = vector_store.as_retriever(search_kwargs={'k': 3})
    return retriever


# FastMCP 서버 초기화 및 구성
mcp = FastMCP(
    "Retriever",
    instructions="데이터베이스에서 청년월세지원 정보를 검색할 수 있는 Retriever입니다.",
)


@mcp.tool()
async def retrieve(query: str) -> str:
    """쿼리를 기반으로 청년월세지원 관련 문서 데이터베이스에서 정보를 검색합니다.

    이 함수는 Retriever를 생성하고, 제공된 쿼리로 검색을 수행한 후,
    검색된 청년월세지원 관련 문서의 내용을 연결하여 반환합니다.

    Args:
        query: 관련 정보를 찾기 위한 검색 쿼리

    Returns:
        검색된 청년월세지원 관련 문서의 텍스트 내용을 연결한 문자열
    """
    retriever = create_retriever()

    # invoke() 메서드를 사용하여 쿼리 기반의 관련 문서를 가져옵니다
    retrieved_docs = retriever.invoke(query)

    # 모든 문서 내용을 줄바꿈으로 연결하여 단일 문자열로 반환합니다
    return "\n".join([doc.page_content for doc in retrieved_docs])


if __name__ == "__main__":
    mcp.run(transport="stdio")
