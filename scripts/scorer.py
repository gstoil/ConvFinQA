from typing import Dict

import nltk
from rouge_score import rouge_scorer


class Scorer:
    """Applies common text matching metrics for language generation evaluations."""

    def __init__(self) -> None:
        self._rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    @staticmethod
    def levenshtein_sim(s1: str, s2: str) -> float:
        dist = nltk.edit_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        return 1 - dist / max_len if max_len != 0 else 1

    @staticmethod
    def normalise_as_num(s: str):
        """
        Try to normalise values as numbers and if value returned as percentage try to change it to number in order
        to avoid superficial differences like 18% vs 0.18.
        """
        percent = 1
        try:
            if '%' in s:
                s = s.replace('%', '')
                percent = 100
            s = float(s)
            return str(s / percent)
        except (ValueError, TypeError):
            return s

    def evaluation_metrics(self, expected_ans: str, computed_ans: str) -> Dict[str, float]:
        computed_ans = self.normalise_as_num(computed_ans)
        expected_ans = self.normalise_as_num(expected_ans)

        exact_match = 1 if computed_ans == expected_ans else 0
        rouge_l = self._rouge.score(expected_ans, computed_ans)['rougeL'].fmeasure
        lev_sim = self.levenshtein_sim(expected_ans, computed_ans)
        if expected_ans == 'yes' and computed_ans == '1.0':
            exact_match = 1
            rouge_l = 1
            lev_sim = 1

        return {
            'exact_match': exact_match,
            'lev_sim': lev_sim,
            'rouge': rouge_l,
        }
