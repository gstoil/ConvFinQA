import json
import os
from pathlib import Path
from typing import List

from loguru import logger
from pydantic import BaseModel, RootModel, ConfigDict, field_validator


class Annotation(BaseModel):
    dialogue_break: List[str] = []
    exe_ans_list: List[str] = []
    turn_program: List[str] = []

    model_config = ConfigDict(extra='ignore')

    @field_validator('exe_ans_list', mode='before')
    @classmethod
    def clean_exe_ans(cls, v):
        cleaned = []
        for item in v or []:
            try:
                num = float(item)
                num_to_add = str(int(num)) if num.is_integer() else str(num)
                cleaned.append(num_to_add)
            except (ValueError, TypeError):
                # Non-numeric like "yes"
                cleaned.append(str(item))
        return cleaned


class ParsedItem(BaseModel):
    pre_text: str
    post_text: str
    table_ori: List[List[str]]
    id: str
    annotation: Annotation = Annotation()

    model_config = ConfigDict(extra='ignore')

    @field_validator('pre_text', 'post_text', mode='before')
    @classmethod
    def join_text(cls, v):
        if isinstance(v, list):
            return ' '.join(v).strip()
        return v

    @property
    def dialogue_break(self) -> List[str]:
        return self.annotation.dialogue_break

    @property
    def exe_ans_list(self) -> List[str]:
        return self.annotation.exe_ans_list

    @property
    def turn_program(self) -> List[str]:
        return self.annotation.turn_program


class ParsedDataset(RootModel[List[ParsedItem]]):
    pass


class ConvFinQaOriginalLoader:
    train_file = 'data/train.json'

    def __init__(self, dataset_file: str = None):
        self.dataset_file = dataset_file if dataset_file else self.train_file
        logger.info(f'Loading dataset from {self.dataset_file}')
        with open(os.path.join(Path.cwd(), self.dataset_file), 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            self.financial_dataset = ParsedDataset.model_validate(dataset).root

    def find_document(self, document_id) -> ParsedItem:
        found_doc = next(
            (doc for doc in self.financial_dataset if doc.id == document_id),
            None,
        )
        if not found_doc:
            raise ValueError(f'Document {document_id} not found in financial dataset')
        return found_doc
