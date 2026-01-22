# parallel_node_graph.py
# 청년월세지원 조건 확인을 위한 병렬 처리 그래프
# 다른 파일에서 graph를 import하여 사용할 수 있습니다.
# 사용 예시: from parallel_node_graph import graph

from dotenv import load_dotenv
load_dotenv()

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import TavilySearchResults


# ============================================================
# 1. State 정의
# ============================================================
class AgentState(TypedDict):
    query: str  # 사용자 질문
    answer: str  # 최종 답변
    month_rent_condition_information: str  # 청년월세지원 조건 정보
    target_name: str  # 청년월세지원 대상자 이름
    target_information: str  # 청년월세지원 대상자 정보


# ============================================================
# 2. Vector Store 및 Retriever 설정
# ============================================================
embedding_function = OpenAIEmbeddings(model='text-embedding-3-large')

vector_store = Chroma(
    embedding_function=embedding_function,
    collection_name='chroma_collection',
    persist_directory='./chroma_collection'
)
retriever = vector_store.as_retriever(search_kwargs={'k': 1})


# ============================================================
# 3. LLM 및 프롬프트 설정
# ============================================================
rag_prompt = hub.pull('rlm/rag-prompt')
llm = ChatOpenAI(model="gpt-4o-mini")


# ============================================================
# 4. 체인 정의
# ============================================================
# 월세 조건 조회 체인
month_rent_retrieval_chain = (
    {'context': retriever, 'question': RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

month_rent_condition_prompt = ChatPromptTemplate.from_messages([
    ('system', '사용자의 입력에서 청년월세지원 조건을 깔끔하게 정리해주세요.'),
    ('human', '{month_rent_condition_information}')
])

month_rent_condition_chain = (
    {'month_rent_condition_information': RunnablePassthrough()}
    | month_rent_condition_prompt
    | llm
    | StrOutputParser()
)

month_rent_chain = {'month_rent_equation_information': month_rent_retrieval_chain} | month_rent_condition_chain

# 대상자 정보 조회를 위한 Tavily 검색 도구
tavily_search_tool = TavilySearchResults(
    max_results=1,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
    include_images=False,
)

# 대상자 정보 추출 프롬프트
tax_market_ratio_prompt = ChatPromptTemplate.from_messages([
    ('system', '아래 정보를 기반으로 대상자의 나이, 소득, 주택 소유 여부를 판단해주세요.\n\nContext:\n{context}'),
    ('human', '{query}')
])

# 최종 답변 프롬프트
final_answer_prompt = ChatPromptTemplate.from_messages([
    ('system', '''당신은 청년월세지원 담당자입니다. 아래 문서를 참고해서 사용자의 질문에 대한 청년월세지원 신청 가능 여부를 판단해주세요.

청년월세지원 신청 조건:{context}'''),
    ('human', '''다음 대상자의 청년월세신청 가능여부를 판별해주세요.

대상자이름: {target_name}
대상자 정보: {target_information}''')
])


# ============================================================
# 5. 노드 함수 정의
# ============================================================
def get_target_extract(query: str) -> str:
    """사용자의 질문에서 대상자 이름을 추출하는 함수"""
    prompt = ChatPromptTemplate.from_messages([
        ('system', '사용자의 질문에서 언급되고 있는 사람의 이름을 추출해주세요. 추가적인 문장없이 이름만 추출해주세요.'),
        ('human', '{query}')
    ])
    chain = prompt | llm | StrOutputParser()
    target_name = chain.invoke({'query': query})
    return target_name


def get_month_rent_condition(state: AgentState) -> AgentState:
    """청년월세지원 조건 정보를 조회하는 노드"""
    month_rent_condition_question = '청년월세지원 신청조건에 대한 내용을 정리해주세요.'
    month_rent_condition = month_rent_chain.invoke(month_rent_condition_question)
    return {'month_rent_condition_information': month_rent_condition}


def get_target_information(state: AgentState) -> AgentState:
    """대상자의 기본 정보를 검색하는 노드"""
    target_name = get_target_extract(state['query'])
    query = f'{target_name}의 기본정보(나이, 소득, 주택 소유 여부)를 검색해주세요.'

    # tavily_search_tool을 사용하여 쿼리를 실행하고 컨텍스트를 얻습니다.
    context = tavily_search_tool.invoke(query)

    target_information_chain = (
        tax_market_ratio_prompt
        | llm
        | StrOutputParser()
    )

    target_information = target_information_chain.invoke({'context': context, 'query': query})

    return {'target_information': target_information, 'target_name': target_name}


def final_answer_generation(state: AgentState) -> AgentState:
    """최종 답변을 생성하는 노드"""
    month_rent_condition_information = state['month_rent_condition_information']
    target_name = state['target_name']
    target_information = state['target_information']

    final_answer_chain = (
        final_answer_prompt
        | llm
        | StrOutputParser()
    )

    final_answer = final_answer_chain.invoke({
        'context': month_rent_condition_information,
        'target_name': target_name,
        'target_information': target_information,
    })

    return {'answer': final_answer}


# ============================================================
# 6. Graph 빌드 및 컴파일
# ============================================================
graph_builder = StateGraph(AgentState)

# 노드 추가
graph_builder.add_node('get_month_rent_condition', get_month_rent_condition)
graph_builder.add_node('get_target_information', get_target_information)
graph_builder.add_node('final_answer_generation', final_answer_generation)

# 엣지 연결 (병렬 처리)
# START에서 두 노드로 동시에 분기하여 병렬 처리
graph_builder.add_edge(START, 'get_month_rent_condition')
graph_builder.add_edge(START, 'get_target_information')

# 두 노드가 완료되면 final_answer_generation으로 합류
graph_builder.add_edge('get_month_rent_condition', 'final_answer_generation')
graph_builder.add_edge('get_target_information', 'final_answer_generation')

# final_answer_generation 완료 후 종료
graph_builder.add_edge('final_answer_generation', END)

# 그래프 컴파일
graph = graph_builder.compile()