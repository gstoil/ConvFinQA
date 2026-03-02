import os

from openai import OpenAI
from pydantic import BaseModel, Field


class Response(BaseModel):
    answer: float = Field(
        description='The answer to the user question. It needs to be a float even if question '
        'is asking for some percentage.'
    )
    reason: str = Field(description='The reason for your answer', default='')


class LLMInference:
    def __init__(self, model):
        api_key = os.environ.get('OPENAI_API_KEY')
        self.llm_client = OpenAI(api_key=api_key)
        self.model = model

    def answer_question(self, messages, temperature=0.0) -> Response:
        response = self.llm_client.chat.completions.parse(
            model=self.model,
            messages=messages,
            temperature=temperature,
            response_format=Response,
        )
        return response.choices[0].message.parsed
