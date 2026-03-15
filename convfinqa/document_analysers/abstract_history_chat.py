from abc import ABCMeta, abstractmethod
from data_loaders.convfinqa_original_loader import ParsedItem
from llm_client import LLMInference, Response


class HistoryBasedChat(metaclass=ABCMeta):
    registry = {}

    def __init__(self, document: ParsedItem, model):
        self.model = model
        self.llm_inference = LLMInference(model)
        self.history = []
        self.document = document

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

    @abstractmethod
    def run_single_turn(self, message) -> Response:
        pass

    def update_history(self, message, response) -> None:
        self.history.extend(
            [
                {'role': 'user', 'content': message},
                {'role': 'assistant', 'content': response},
            ]
        )
