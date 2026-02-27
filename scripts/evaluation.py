import argparse
import concurrent.futures
from datetime import datetime
import json
import random
from collections import defaultdict

import dotenv
from tqdm import tqdm

from chat_with_history import HistoryBasedChat
from data_loaders.convfinqa_original_loader import ConvFinQaOriginalLoader
from scripts.scorer import Scorer


dotenv.load_dotenv('.env')

random.seed(42)


def compute_avg_scores(results):
    totals = defaultdict(float)
    for turn_result in results:
        for key, value in turn_result['metrics'].items():
            totals[key] += value
    return {key: value / len(results) for key, value in totals.items()}


def run_complete_test(executor_name, doc_as_string, test_case, model, scorer):
    """Run all questions over a specific document."""
    chat_with_data = HistoryBasedChat.create(executor_name, document_as_string=doc_as_string, model=model)
    test_results = {
        'detailed_results': list(),
        'turn_program': test_case.turn_program,
        'avg_scores': dict(),
    }
    for i, single_question in enumerate(test_case.dialogue_break):
        response = chat_with_data.run_single_turn(single_question)

        expected_ans = test_case.exe_ans_list[i]
        metrics = scorer.evaluation_metrics(expected_ans, str(response.answer))
        test_results['detailed_results'].append(
            {
                'question': single_question,
                'expected': expected_ans,
                'returned': f'{response.answer} / REASON: {response.reason}',
                'metrics': metrics,
            }
        )
    return test_results


def eval_conv_fin_qa(executor_name, model, file_name=None, sample_size=None):
    data_loader = ConvFinQaOriginalLoader(file_name)
    sample_tests = data_loader.financial_dataset
    if sample_size:
        sample_tests = random.sample(data_loader.financial_dataset, sample_size)
    scorer = Scorer()

    results_report = dict()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_objs = {
            executor.submit(
                run_complete_test,
                executor_name,
                data_loader.format_document(test_case),
                test_case,
                model,
                scorer,
            ): test_case.id
            for test_case in sample_tests
        }
        for future_obj in tqdm(concurrent.futures.as_completed(future_objs), total=len(future_objs)):
            test_results = future_obj.result()
            test_id = future_objs[future_obj]
            results_report[test_id] = test_results
            results_report[test_id]['avg_scores'] = compute_avg_scores(test_results['detailed_results'])

    results_report = dict(sorted(results_report.items()))

    # overall average scores
    totals = defaultdict(float)
    for doc_id, results in results_report.items():
        for key, value in results['avg_scores'].items():
            totals[key] += value
    results_report['total_avg_scores'] = {key: value / len(results_report) for key, value in totals.items()}

    with open(
        f'scripts/results_{datetime.today().strftime("%Y_%m_%d-%H_%M_%S")}',
        'w',
        encoding='utf-8',
    ) as f:
        json.dump(results_report, f, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', help='File to use', type=str)
    parser.add_argument(
        '--model',
        '-m',
        help='OpenAI model to use',
        type=str,
        default='gpt-4.1-mini',
    )
    parser.add_argument(
        '--executor',
        '-e',
        help='Executor approach',
        choices=HistoryBasedChat.registry.keys(),
        required=True,
    )
    parser.add_argument('--sample', '-s', help='Sample size', type=int)
    args = parser.parse_args()
    eval_conv_fin_qa(args.executor, args.model, args.file, args.sample)
