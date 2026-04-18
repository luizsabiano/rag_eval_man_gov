import json
import os
import numpy as np

from src import directories

def summarize_rag_results():

    rag_eval_results_dir = directories.RAG_EVAL_RESULTS_PATH

    list_dataset_names = os.listdir(rag_eval_results_dir)

    print('\nSummarizing Results')

    list_metrics = [
        'faithfulness',
        'answer_relevancy',
    ]

    for dataset_name in list_dataset_names:

        print(f'\n\tDataset: {dataset_name}')

        dataset_results_dir = os.path.join(rag_eval_results_dir, dataset_name)

        list_results_file_names = os.listdir(dataset_results_dir)

        if len(list_results_file_names) == 0:
            continue

        list_results_file_names = [document_name for document_name in list_results_file_names
                                   if document_name.endswith('.json')]

        csv_content = None

        for results_file_name in list_results_file_names:

            print(f'\n\t\tOutput: {results_file_name}\n')

            if csv_content is None:
                csv_content = 'Model;'
                eval_metrics = {}
                for metric_name in list_metrics:
                    csv_content += f'{metric_name};'
                    eval_metrics[metric_name] = []
                csv_content = csv_content[:-1] + '\n'

            results_file_path = os.path.join(dataset_results_dir, results_file_name)

            with open(file=results_file_path, mode='r', encoding='utf-8') as file:
                list_eval_data = json.load(file)

            results_file_name = results_file_name.replace('.json', '')

            for eval_data in list_eval_data:
                for key in eval_metrics:
                    value = eval_data[key]
                    if value is None:
                        value = 0.0
                    eval_metrics[key].append(value)

            csv_content += f'{results_file_name};'

            for metric_name in list_metrics:
                mean_value = np.mean(eval_metrics[metric_name])
                std_value = np.std(eval_metrics[metric_name])
                print(f'\t\t\t{metric_name}: {mean_value:.3f} ~ {std_value:.3f}')

                csv_content += f'{mean_value:.3f} ({std_value:.3f});'

            csv_content = csv_content[:-1]

            csv_content += '\n'

        results_file_path = os.path.join(dataset_results_dir, 'summary_results.csv')

        with open(file=results_file_path, mode='w', encoding='utf-8') as results_file_path:
            results_file_path.write(csv_content)