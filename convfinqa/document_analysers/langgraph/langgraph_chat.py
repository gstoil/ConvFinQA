from abc import abstractmethod
from typing import TypedDict, Optional, List

from document_analysers.abstract_history_chat import HistoryBasedChat
from llm_client import Response


class QAItem(TypedDict):
    question: str
    answer: str
    reason: str


class FinancialState(TypedDict):
    question: str

    table_answer: Optional[dict]
    text_answer: Optional[dict]
    final_answer: Optional[dict]

    route: Optional[str]

    history: List[QAItem]


class LangGraphChatter(HistoryBasedChat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.app = self.build_graph()

    @abstractmethod
    def build_graph(self):
        pass

    def run_single_turn(self, message) -> Response:
        config = {'configurable': {'thread_id': self.document.id}}
        result = self.app.invoke({'question': message}, config=config)
        return Response(**result['final_answer'])
