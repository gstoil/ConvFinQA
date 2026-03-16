import argparse
from collections import defaultdict
from typing import Dict, List

import tiktoken

from convfinqa.data_loaders.convfinqa_original_loader import ParsedItem, ConvFinQaOriginalLoader


def longest_doc(financial_dataset) -> ParsedItem:
    docs_to_lengths = [
        (
            len(conv_fin_record.pre_text) + len(conv_fin_record.post_text) + len(str(conv_fin_record.table_json)),
            conv_fin_record,
        )
        for conv_fin_record in financial_dataset
    ]
    return sorted(docs_to_lengths, key=lambda x: x[0], reverse=True)[0][1]


def find_longest_dialogue(financial_dataset: List[ParsedItem]) -> ParsedItem:
    docs_to_lengths = [(len(conv_fin_record.dialogue_break), conv_fin_record) for conv_fin_record in financial_dataset]
    return sorted(docs_to_lengths, key=lambda x: x[0], reverse=True)[0][1]


def is_not_number(s: str) -> bool:
    try:
        float(s)
        return False
    except ValueError:
        return True


def non_numerical_expectations(financial_dataset: List[ParsedItem]) -> Dict[str, List[str]]:
    docs_with_non_numerical_expectations = defaultdict(list)
    for conv_fin_record in financial_dataset:
        expectations = conv_fin_record.exe_ans_list
        for expectation in expectations:
            if is_not_number(expectation):
                docs_with_non_numerical_expectations[conv_fin_record.id].append(expectation)

    return docs_with_non_numerical_expectations


def analyse_data(file_name):

    data_loader = ConvFinQaOriginalLoader(file_name)
    longest_document = longest_doc(data_loader.financial_dataset)
    encoding = tiktoken.encoding_for_model('gpt-4o')
    tokens = encoding.encode(str(longest_document))
    print(f'Longest document contains: {len(tokens)} tokens')

    print(non_numerical_expectations(data_loader.financial_dataset))

    longest_dialogue = find_longest_dialogue(data_loader.financial_dataset)
    print(f'Longest dialogue is {longest_dialogue.id} and has {len(longest_dialogue.dialogue_break)} turns')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', help='File to use', type=str, default='data/train.json')
    args = parser.parse_args()

    analyse_data(args.file)
