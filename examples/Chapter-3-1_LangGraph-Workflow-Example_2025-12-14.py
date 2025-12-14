"""
Chapter 3-1: LangGraph 워크플로우 실전 예제
학습 날짜: 2025-12-14

이 예제는 간단한 고객 문의 처리 에이전트를 구현합니다.
고객의 질문을 받아 적절한 카테고리로 분류하고, 답변을 생성하는 과정을 보여줍니다.
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage


# ============================================================
# 1단계: 문제 정의
# ============================================================
# 문제: 고객 문의를 받아 카테고리를 분류하고 적절한 답변을 생성하는 에이전트 구축
# 목표:
#   - 고객 문의를 "기술지원", "결제", "일반문의"로 자동 분류
#   - 각 카테고리에 맞는 답변 생성
#   - 불명확한 문의는 재질문 유도


# ============================================================
# 2단계: 상태 설계
# ============================================================
# 에이전트가 작업하는 동안 필요한 모든 정보를 담는 구조 정의

class AgentState(TypedDict):
    """
    에이전트의 상태를 정의하는 클래스

    - question: 고객의 원래 질문
    - category: 분류된 카테고리 ("기술지원", "결제", "일반문의", "불명확")
    - answer: 생성된 최종 답변
    - messages: 대화 기록 (선택적)
    """
    question: str
    category: str
    answer: str
    messages: Annotated[list, "대화 메시지 리스트"]


# ============================================================
# 3단계: 노드 구현
# ============================================================
# 각 작업 단위를 함수로 구현

def classify_question(state: AgentState) -> AgentState:
    """
    노드 1: 질문 분류하기

    고객의 질문을 분석하여 적절한 카테고리로 분류합니다.
    실제 구현에서는 LLM을 사용하지만, 여기서는 키워드 기반으로 간단히 구현합니다.
    """
    question = state["question"].lower()

    # 간단한 키워드 기반 분류 (실제로는 LLM 사용)
    if any(keyword in question for keyword in ["오류", "버그", "작동", "안됨", "에러"]):
        category = "기술지원"
    elif any(keyword in question for keyword in ["결제", "환불", "가격", "구매", "요금"]):
        category = "결제"
    elif any(keyword in question for keyword in ["안녕", "문의", "정보", "알려"]):
        category = "일반문의"
    else:
        category = "불명확"

    # 상태 업데이트
    state["category"] = category
    state["messages"].append(HumanMessage(content=f"질문 분류 완료: {category}"))

    return state


def generate_technical_answer(state: AgentState) -> AgentState:
    """
    노드 2-A: 기술지원 답변 생성

    기술지원 카테고리에 해당하는 답변을 생성합니다.
    """
    answer = f"""
    기술지원팀 답변:

    문의하신 기술적 문제에 대해 도움을 드리겠습니다.
    원활한 지원을 위해 다음 정보를 제공해 주세요:

    1. 사용 중인 기기 및 운영체제
    2. 오류 발생 시점 및 재현 방법
    3. 오류 메시지 캡처 화면

    추가 문의사항이 있으시면 support@example.com으로 연락 주시기 바랍니다.
    """

    state["answer"] = answer
    state["messages"].append(AIMessage(content="기술지원 답변 생성 완료"))

    return state


def generate_payment_answer(state: AgentState) -> AgentState:
    """
    노드 2-B: 결제 관련 답변 생성

    결제/환불 카테고리에 해당하는 답변을 생성합니다.
    """
    answer = f"""
    결제팀 답변:

    결제 관련 문의에 답변드립니다.

    - 환불 정책: 구매 후 7일 이내 전액 환불 가능
    - 결제 수단: 신용카드, 계좌이체, 간편결제 지원
    - 영수증 발급: 마이페이지 > 구매내역에서 확인 가능

    추가 문의는 billing@example.com으로 연락 주시기 바랍니다.
    """

    state["answer"] = answer
    state["messages"].append(AIMessage(content="결제 답변 생성 완료"))

    return state


def generate_general_answer(state: AgentState) -> AgentState:
    """
    노드 2-C: 일반 문의 답변 생성

    일반적인 문의에 대한 답변을 생성합니다.
    """
    answer = f"""
    고객지원팀 답변:

    안녕하세요, 문의해 주셔서 감사합니다.

    저희 서비스에 대한 자세한 정보는 다음에서 확인하실 수 있습니다:
    - 홈페이지: www.example.com
    - FAQ: www.example.com/faq
    - 고객센터: 1588-1234 (평일 09:00-18:00)

    추가 도움이 필요하시면 언제든지 문의해 주세요.
    """

    state["answer"] = answer
    state["messages"].append(AIMessage(content="일반 답변 생성 완료"))

    return state


def request_clarification(state: AgentState) -> AgentState:
    """
    노드 2-D: 재질문 요청

    질문이 불명확한 경우 추가 정보를 요청합니다.
    """
    answer = f"""
    문의 내용이 명확하지 않아 정확한 답변을 드리기 어렵습니다.

    다음 중 해당하는 항목을 선택하여 다시 문의해 주시겠어요?

    1. 기술적 문제 (오류, 버그, 작동 이슈)
    2. 결제 관련 (환불, 구매, 요금)
    3. 서비스 일반 정보

    구체적으로 어떤 부분이 궁금하신지 말씀해 주시면 더 정확한 답변을 드리겠습니다.
    """

    state["answer"] = answer
    state["messages"].append(AIMessage(content="재질문 요청 생성 완료"))

    return state


# ============================================================
# 4단계: 에지 연결 (조건부 라우팅)
# ============================================================
# 상태에 따라 다음에 실행할 노드를 결정하는 함수

def route_question(state: AgentState) -> Literal["technical", "payment", "general", "unclear"]:
    """
    분류된 카테고리에 따라 다음 노드를 결정합니다.

    이 함수는 에지의 조건부 분기를 구현합니다.
    상태의 category 값에 따라 어느 답변 생성 노드로 갈지 결정합니다.
    """
    category = state["category"]

    if category == "기술지원":
        return "technical"
    elif category == "결제":
        return "payment"
    elif category == "일반문의":
        return "general"
    else:
        return "unclear"


# ============================================================
# 5단계: 그래프 컴파일
# ============================================================
# 노드와 에지를 연결하여 전체 워크플로우를 구성

def create_customer_support_agent():
    """
    고객 지원 에이전트 그래프를 생성하고 컴파일합니다.

    그래프 구조:
    START -> classify_question -> route_question -> [답변 노드들] -> END
    """
    # StateGraph 초기화
    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("classify", classify_question)
    workflow.add_node("technical", generate_technical_answer)
    workflow.add_node("payment", generate_payment_answer)
    workflow.add_node("general", generate_general_answer)
    workflow.add_node("unclear", request_clarification)

    # 시작점 설정: 항상 classify 노드에서 시작
    workflow.set_entry_point("classify")

    # 조건부 에지 추가: classify 노드 다음에 route_question 함수로 분기
    workflow.add_conditional_edges(
        "classify",  # 출발 노드
        route_question,  # 라우팅 함수
        {
            # 라우팅 함수의 반환값 -> 목적지 노드 매핑
            "technical": "technical",
            "payment": "payment",
            "general": "general",
            "unclear": "unclear"
        }
    )

    # 각 답변 노드에서 END로 가는 에지 추가
    workflow.add_edge("technical", END)
    workflow.add_edge("payment", END)
    workflow.add_edge("general", END)
    workflow.add_edge("unclear", END)

    # 그래프 컴파일
    app = workflow.compile()

    return app


# ============================================================
# 6단계: 실행 및 테스트
# ============================================================

def test_agent():
    """
    에이전트를 다양한 질문으로 테스트합니다.
    """
    # 에이전트 생성
    agent = create_customer_support_agent()

    # 테스트 케이스들
    test_questions = [
        "로그인이 안되는 오류가 발생했어요",
        "환불은 어떻게 하나요?",
        "서비스 이용 시간이 궁금합니다",
        "그냥 안녕하세요?"
    ]

    print("=" * 60)
    print("고객 지원 에이전트 테스트")
    print("=" * 60)

    for question in test_questions:
        print(f"\n질문: {question}")
        print("-" * 60)

        # 초기 상태 설정
        initial_state = {
            "question": question,
            "category": "",
            "answer": "",
            "messages": []
        }

        # 에이전트 실행
        result = agent.invoke(initial_state)

        # 결과 출력
        print(f"분류: {result['category']}")
        print(f"\n답변:\n{result['answer']}")
        print("=" * 60)


# ============================================================
# 실행 예제
# ============================================================

if __name__ == "__main__":
    """
    이 스크립트를 실행하면 다양한 고객 문의에 대해
    에이전트가 어떻게 분류하고 답변하는지 확인할 수 있습니다.

    실행 방법:
    $ python Chapter-3-1_LangGraph-Workflow-Example_2025-12-14.py

    주의: 실제 실행을 위해서는 langgraph와 langchain 패키지가 필요합니다.
    $ pip install langgraph langchain-core
    """
    test_agent()


# ============================================================
# 학습 포인트 정리
# ============================================================
"""
이 예제에서 배울 수 있는 핵심 개념:

1. 상태 설계 (AgentState)
   - TypedDict를 사용하여 타입 안정성 확보
   - 에이전트가 필요한 모든 정보를 구조화

2. 노드 구현 (함수들)
   - 각 노드는 상태를 받아서 처리하고 업데이트된 상태를 반환
   - 단일 책임 원칙: 각 노드는 하나의 명확한 작업만 수행

3. 조건부 라우팅 (route_question)
   - 상태에 따라 다음 행동을 결정
   - Literal 타입을 사용하여 가능한 경로를 명시

4. 그래프 구성
   - add_node: 작업 단위 추가
   - add_edge: 고정된 흐름 정의
   - add_conditional_edges: 조건부 분기 정의
   - set_entry_point: 시작점 지정

5. 실행 및 테스트
   - invoke()로 초기 상태를 전달하여 실행
   - 결과 상태를 확인하여 디버깅 및 개선

이러한 구조적 접근을 통해 복잡한 에이전트 로직도
예측 가능하고 유지보수 가능하게 구현할 수 있습니다.
"""
