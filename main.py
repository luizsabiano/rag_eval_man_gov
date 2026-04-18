import os

from src import  collections_, llm_, evaluation_measures, directories
from dotenv import load_dotenv

load_dotenv()

SABIA_API_KEY = os.getenv('SABIA_API_KEY')


def prepare_the_environment(list_splitting_parameters_: list):
    # get the chunks in /data/chunks and save on vector database ChromaDB
    collections_.insert_chromadb_content(list_splitting_parameters_)


def run_metrics(list_splitting_parameters_: list, is_run_similarity_metrics: bool,
                is_run_ragas_metrics: bool, experiments_:list):

    if is_run_similarity_metrics or is_run_ragas_metrics:

        evaluation_measures.compute_eval_measures(
            list_splitting_parameters_,
            is_run_similarity_metrics,
            is_run_ragas_metrics,
            experiments_
        )

    # Sintetiza a média e o desvio padrão dos resultados

    evaluation_measures.summarize_eval_results(directories.EVAL_RESULTS_PATH)


def run_experiment(experiments_: list, list_splitting_parameters_: list, n_samples_: int = -1,
                   top_k_retrieval_: int = 5, is_prepare_environment_: bool = True,
                   is_run_generation: bool = True,
                   is_run_metrics: bool = True, is_run_similarity_metrics: bool = True,
                   is_run_ragas_metrics: bool = True):

    if is_prepare_environment_:

        print('\n\tPreparing Environment')

        prepare_the_environment(list_splitting_parameters_)

    if is_run_generation:

        print('\n\tRunning Answers Generation')

        # Submete as perguntas aos LLMs

        llm_.do_query_to_llm(
            list_splitting_parameters_,
            experiments_,
            n_samples_,
            top_k_retrieval_
        )

    if is_run_metrics:

        print('\n\tRunning Evaluation Metrics')

        run_metrics(
            list_splitting_parameters_,
            is_run_similarity_metrics,
            is_run_ragas_metrics,
            experiments_
        )


if __name__ == '__main__':

    experiments = [
        'chunk_size',
        'rewrite',
        'hyde',
        'multiple_queries'
    ]

    list_splitting_parameters = [
        500,
        1000,
        2000,
        4000,
        8000,
        'semantic'
    ]

    n_samples = -1

    top_k_retrieval = 5

    print(f'\n{"=" * 50} Running Experiments {"=" * 50}')

    run_experiment(
        experiments_=experiments,
        list_splitting_parameters_=list_splitting_parameters,
        n_samples_=n_samples,
        top_k_retrieval_=top_k_retrieval,
        is_prepare_environment_=True,
        is_run_generation=True,
        is_run_metrics=True,
        is_run_similarity_metrics=True,
        is_run_ragas_metrics=True
    )