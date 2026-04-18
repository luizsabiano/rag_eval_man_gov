from evaluate import load
from bert_score import BERTScorer


def compute_bertscore(list_reference_answer_, list_generated_answer_):
    scorer = BERTScorer(lang='pt', rescale_with_baseline=False)
    precision, recall, f1_score = scorer.score(
        list_generated_answer_,
        list_reference_answer_,
        verbose=False
    )
    return {
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1_score': f1_score.mean().item()
    }


def compute_rouge(list_reference_answer_, list_generated_answer_):
    metric = load('rouge')
    results = metric.compute(predictions=list_generated_answer_,
                             references=list_reference_answer_,
                             use_stemmer=False)
    return results
