from data_loaders.convfinqa_original_loader import (
    ConvFinQaOriginalLoader,
    ParsedItem,
)
import tiktoken


def longest_doc(financial_dataset) -> ParsedItem:
    docs_to_lengths = [
        (
            len(conv_fin_record.pre_text)
            + len(conv_fin_record.post_text)
            + len(str(ConvFinQaOriginalLoader.table_to_json(conv_fin_record.table_ori))),
            conv_fin_record,
        )
        for conv_fin_record in financial_dataset
    ]
    return sorted(docs_to_lengths, key=lambda x: x[0], reverse=True)[0][1]


def longest_dialogue(financial_dataset) -> ParsedItem:
    docs_to_lengths = [(len(conv_fin_record.dialogue_break), conv_fin_record) for conv_fin_record in financial_dataset]
    return sorted(docs_to_lengths, key=lambda x: x[0], reverse=True)[0][1]


data_loader = ConvFinQaOriginalLoader()
longest_document = longest_doc(data_loader.financial_dataset)
encoding = tiktoken.encoding_for_model('gpt-4o')
tokens = encoding.encode(str(longest_document))
print(f'Longest document contains: {len(tokens)} tokens')


longest_dialogue = longest_dialogue(data_loader.financial_dataset)
print(f'Longest dialogue is {longest_dialogue.id} and has {len(longest_dialogue.dialogue_break)} turns')
