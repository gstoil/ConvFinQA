import json
import os
from pathlib import Path
from typing import List
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
    financial_dataset = 'data/train.json'

    def __init__(self):
        with open(
            os.path.join(Path.cwd(), self.financial_dataset),
            'r',
            encoding='utf-8',
        ) as f:
            dataset = json.load(f)
            self.financial_dataset = ParsedDataset.model_validate(dataset).root

    @staticmethod
    def table_to_json(table: list[list[str]]) -> dict:
        if len(table) < 2:
            return {}

        headers = [h.strip() for h in table[0][1:]]

        def parse_number(value: str) -> float:
            value = value.strip()
            if value.startswith('(') and value.endswith(')'):
                value = '-' + value[1:-1]
            value = value.replace('$', '').replace(',', '')
            return float(value)

        result = {header: {} for header in headers}

        for row in table[1:]:
            metric = row[0].strip().lower()
            for i, header in enumerate(headers):
                if i + 1 < len(row):
                    try:
                        result[header][metric] = parse_number(row[i + 1])
                    except Exception:
                        continue

        return result

    def format_document(self, data_record: ParsedItem) -> str:
        return (
            data_record.pre_text
            + '\n'
            + f'<table>\n{str(self.table_to_json(data_record.table_ori))}\n</table>\n'
            + data_record.post_text
        )

    def find_document(self, document_id) -> ParsedItem:
        found_doc = next(
            (doc for doc in self.financial_dataset if doc.id == document_id),
            None,
        )
        if not found_doc:
            raise ValueError(f'Document {document_id} not found in financial dataset')
        return found_doc
