from abc import ABCMeta, abstractmethod
from typing import List

from data_loaders.convfinqa_original_loader import ParsedItem
from llm_client import LLMInference, Response


class HistoryBasedChat(metaclass=ABCMeta):
    registry = {}

    def __init__(self, document: ParsedItem, model):
        self.llm_inference = LLMInference(model)
        self.history = []
        self.document = document

    @abstractmethod
    def build_messages(self, message) -> List:
        pass

    @classmethod
    def create(cls, name, **kwargs):
        if name.lower() in cls.registry:
            return cls.registry[name.lower()](**kwargs)
        else:
            raise ValueError(f'Unknown product type: {name}')

    @classmethod
    def register(cls, name):
        """Register complete classes"""

        def inner(subclass):
            cls.registry[name.lower()] = subclass
            return subclass

        return inner

    def run_single_turn(self, message) -> Response:
        """Uses LLM to answer a single question based on past history and then records answer to history"""
        messages = self.build_messages(message)
        response = self.llm_inference.answer_question(messages)
        self.update_history(message, response.answer)
        return response

    def update_history(self, message, response) -> None:
        self.history.extend(
            [
                {'role': 'user', 'content': message},
                {'role': 'assistant', 'content': response},
            ]
        )
