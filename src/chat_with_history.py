from abc import ABCMeta, abstractmethod
from typing import List
from prompts import system_prompt_default, user_prompt
from llm_client import LLMInference, Response


class HistoryBasedChat(metaclass=ABCMeta):
    registry = {}

    def __init__(self, document_as_string, model):
        self.llm_inference = LLMInference(model)
        self.history = []
        self.system_prompt_compiled = system_prompt_default.format(report=document_as_string)

    @abstractmethod
    def build_messages(self, message) -> List:
        pass

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


@HistoryBasedChat.register('openai_history_style')
class OpenAIStyleHistoryChat(HistoryBasedChat):
    def build_messages(self, message):
        return (
            [{'role': 'system', 'content': self.system_prompt_compiled}]
            + self.history
            + [
                {
                    'role': 'user',
                    'content': user_prompt.format(question=message),
                }
            ]
        )


@HistoryBasedChat.register('embedded_history_style')
class EmbeddedHistoryChat(HistoryBasedChat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history_addition = """Here is the conversation history so far:
        <history>
        {history}
        </history>
        """

    def build_messages(self, message):
        history_str = '\n'.join([f'{msg["role"]}: {msg["content"]}' for msg in self.history])
        history_compiled = self.history_addition.format(history=history_str)
        return [
            {
                'role': 'system',
                'content': self.system_prompt_compiled + history_compiled,
            }
        ] + [{'role': 'user', 'content': user_prompt.format(question=message)}]
