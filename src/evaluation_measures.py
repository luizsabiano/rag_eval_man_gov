import os
import json
import sys
import numpy as np

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from src import directories, similarity_measures
from tqdm import tqdm

def compute_eval_measures(list_splitting_parameters: list, is_run_similarity_metrics: bool,
                          is_run_ragas_metrics: bool, experiments_: list):

    load_dotenv()

    list_dataset_names = os.listdir(directories.ANSWERS_PATH)

    openai_api_key = os.getenv('OPENAI_API_KEY')

    client = AsyncOpenAI(
        api_key=openai_api_key,
    )

    llm_evaluator = llm_factory(
        model='gpt-4o-mini',
        client=client,
        # max_tokens=16_384
        max_tokens=8_384
    )

    embeddings_evaluator = embedding_factory(
        provider='openai',
        model='text-embedding-3-small',
        client=client,
        interface='modern'
    )

    answer_relevancy_scorer = AnswerRelevancy(llm=llm_evaluator,
                                              embeddings=embeddings_evaluator)

    faithfulness_scorer = Faithfulness(llm=llm_evaluator)

    for cont_ds, dataset_name in enumerate(list_dataset_names, start=1):

        if dataset_name.split('_')[-1].isdigit():
            chunk_size = int(dataset_name.split('_')[-1])
        else:
            chunk_size = dataset_name.split('_')[-1]

        if chunk_size not in list_splitting_parameters:
            continue

        print(f'\n\t\tDataset {cont_ds}/{len(list_dataset_names)}: {dataset_name}')

        dataset_results_dir = os.path.join(directories.EVAL_RESULTS_PATH, dataset_name)

        os.makedirs(dataset_results_dir, exist_ok=True)

        dataset_outputs_path = os.path.join(directories.ANSWERS_PATH, dataset_name)

        list_outputs_files = os.listdir(dataset_outputs_path)

        for cont, output_file_name in enumerate(list_outputs_files, start=1):

            print(f'\n\t\t\tOutput file {cont} / {len(list_outputs_files)}: {output_file_name}')

            # if ('rewrite' not in output_file_name and 'hyde' not in output_file_name and
            #         'multiple_queries' not in output_file_name and 'chunk_size' not in output_file_name):
            #     continue

            print("experiments name: ", output_file_name.split('_')[-1].split(".")[0])
            # if output_file_name.split('_')[-1].split(".")[0] not in experiments_:
            #     continue


            output_file_path = os.path.join(dataset_outputs_path, output_file_name)

            with open(file=output_file_path, mode='r', encoding='utf-8') as json_file:
                output_data = json.load(json_file)

            results_file_path = os.path.join(dataset_results_dir, output_file_name)

            list_eval_results = []
            dict_eval_results = {}

            if os.path.exists(results_file_path):
                if os.path.exists(results_file_path):
                    with open(file=results_file_path, mode='r', encoding='utf-8') as json_file:
                        eval_data = json.load(json_file)
                    dict_eval_results = {r['id']: r for r in eval_data}

            list_ids = output_data['ids']
            list_questions = output_data['questions']

            list_ground_truth_answers = output_data['ground_truth_answers']
            list_ground_truth_contexts = output_data['ground_truth_contexts']

            list_llm_answers = output_data['llm_answers']

            #Removendo respostas vazias (ex. '') para evitar erros ao executar o RAGAS
            lista_sem_campos_vazios = ["NONE" if item.strip() == ""  else item for item in list_llm_answers]
            list_llm_answers = lista_sem_campos_vazios






            list_retrieved_contexts = output_data['retrieved_contexts']

            with tqdm(total=len(list_ids), file=sys.stdout, colour='blue',
                      desc='\n\t\t\t\tComputing Eval Measures') as pbar:

                for (id_, question, ground_truth_answer, ground_truth_context, llm_answer,
                     retrieved_contexts) in zip(list_ids, list_questions, list_ground_truth_answers,
                                               list_ground_truth_contexts, list_llm_answers,
                                               list_retrieved_contexts):

                    if id_ in dict_eval_results:
                        eval_results = dict_eval_results[id_]
                    else:
                        eval_results = {
                            'id': id_,
                            'question': question,
                            'ground_truth_answer': ground_truth_answer,
                            'ground_truth_context': ground_truth_context,
                            'llm_answer': llm_answer,
                            'retrieved_contexts': retrieved_contexts
                        }

                    llm_answer = llm_answer.replace('\n', ' ').strip()

                    if not isinstance(retrieved_contexts, list):
                        retrieved_contexts = [retrieved_contexts]

                    if is_run_similarity_metrics:

                        if 'bert_metrics' not in eval_results:

                            bert_metrics = similarity_measures.compute_bertscore(
                                list_reference_answer_=[ground_truth_answer],
                                list_generated_answer_=[llm_answer]
                            )

                            eval_results['bert_metrics'] = bert_metrics

                        if 'rouge_metrics' not in eval_results:

                            rouge_metrics = similarity_measures.compute_rouge(
                                list_reference_answer_=[ground_truth_answer],
                                list_generated_answer_=[llm_answer]
                            )

                            eval_results['rouge_metrics'] = rouge_metrics

                    if is_run_ragas_metrics:

                        if 'answer_relevancy' not in eval_results:

                            answer_relevancy_result = answer_relevancy_scorer.score(
                                user_input=question,
                                response=llm_answer
                            )

                            #altera resultado de NaN para 0.0 (NaN - da erro ao abrir Json)
                            eval_results['answer_relevancy'] = answer_relevancy_result.value
                            if np.isnan(answer_relevancy_result.value):
                                eval_results['answer_relevancy'] = 0.0

                        if 'faithfulness' not in eval_results:

                            faithfulness_result = faithfulness_scorer.score(
                                user_input=question,
                                response=llm_answer,
                                retrieved_contexts=retrieved_contexts
                            )

                            eval_results['faithfulness'] = faithfulness_result.value
                            if np.isnan(faithfulness_result.value):
                                eval_results['faithfulness'] = 0.0

                    list_eval_results.append(eval_results)

                    with open(file=results_file_path, mode='w') as outfile:
                        json.dump(list_eval_results, outfile, ensure_ascii=False, indent=4)

                    pbar.update(1)


def summarize_eval_results(eval_results_path_dir: str):

    list_dataset_names = os.listdir(eval_results_path_dir)

    print('\n\tSummarizing Results')

    list_metrics = [
        'faithfulness',
        'answer_relevancy',
        'bert_precision',
        'bert_recall',
        'bert_f1_score',
        'rouge1',
        'rouge2',
        'rougeL',
        'rougeLsum'
    ]

    eval_metrics = {}

    csv_content = None


    if csv_content is None:
        csv_content = 'Model;'
        for metric_name in list_metrics:
            csv_content += f'{metric_name};'
            eval_metrics[metric_name] = []
        csv_content = csv_content[:-1] + '\n'




    for dataset_name in list_dataset_names:

        dataset_results_dir = os.path.join(eval_results_path_dir, dataset_name)

        if not os.path.isdir(dataset_results_dir):
            continue

        print(f'\n\t\tDataset: {dataset_name}')

        list_results_file_names = os.listdir(dataset_results_dir)

        if len(list_results_file_names) == 0:
            continue

        list_results_file_names = [doc_name for doc_name in list_results_file_names
                                   if doc_name.endswith('.json')]

        for results_file_name in list_results_file_names:

            for metric_name in list_metrics:
                eval_metrics[metric_name] = []

            print(f'\n\t\t\tOutput: {results_file_name}\n')

            results_file_path = os.path.join(dataset_results_dir, results_file_name)

            with open(file=results_file_path, mode='r', encoding='UTF-8') as file:
                list_eval_data = json.load(file)

            results_file_name = results_file_name.replace('.json', '')

            for eval_data in list_eval_data:

                if 'bert_metrics' in eval_data:
                    eval_data['bert_precision'] = eval_data['bert_metrics']['precision']
                    eval_data['bert_recall'] = eval_data['bert_metrics']['recall']
                    eval_data['bert_f1_score'] = eval_data['bert_metrics']['f1_score']
                else:
                    eval_data['bert_precision'] = 0
                    eval_data['bert_recall'] = 0
                    eval_data['bert_f1_score'] = 0

                if 'rouge_metrics' in eval_data:
                    eval_data['rouge1'] = eval_data['rouge_metrics']['rouge1']
                    eval_data['rouge2'] = eval_data['rouge_metrics']['rouge2']
                    eval_data['rougeL'] = eval_data['rouge_metrics']['rougeL']
                    eval_data['rougeLsum'] = eval_data['rouge_metrics']['rougeLsum']
                else:
                    eval_data['rouge1'] = 0
                    eval_data['rouge2'] = 0
                    eval_data['rougeL'] = 0
                    eval_data['rougeLsum'] = 0

                for key in eval_metrics:
                    if key in eval_data:
                        value = eval_data[key]
                        if value is None or value == "NaN":
                            value = 0.0
                    else:
                        value = 0
                    eval_metrics[key].append(value)

            csv_content += f'{dataset_name}_{results_file_name};'

            for metric_name in list_metrics:
                mean_value = np.mean(eval_metrics[metric_name])
                std_value = np.std(eval_metrics[metric_name])
                print(f'\t\t\t  {metric_name}: {mean_value:.3f} ~ {std_value:.3f}')


                csv_content += f'{mean_value:.3f} ({std_value:.3f});'

            csv_content = csv_content[:-1]

            csv_content += '\n'


        results_file_path = os.path.join(eval_results_path_dir, 'summary_results.csv')

        with open(file=results_file_path, mode='w', encoding='UTF-8') as results_file_path:
            results_file_path.write(csv_content)