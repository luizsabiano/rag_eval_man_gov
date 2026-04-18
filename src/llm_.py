import os
import sys
import time
import json
import torch

from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoConfig
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src import collections_, rewriter_question, hyde, multiple_queries
from src import directories
from src import get_manual_content
from src import tools
from src.embedding_model import get_embedding_function

load_dotenv()

HF_API_KEY = os.getenv('HF_TOKEN')
SABIA_API_KEY = os.getenv('SABIA_API_KEY')


# Modelos avaliados

LLM_MODELS = [

    {
        'name': 'llama_3_1_8b_it',
        'cloud': 'localhost',
        'model': 'meta-llama/Llama-3.1-8B-Instruct'
    },

    {
        'name': 'gemma_2_9b_it',
        'cloud': 'localhost',
        'model': 'google/gemma-2-9b-it'
    },

    {
        'name': 'sabiazinho_3',
        'checkpoint': 'sabiazinho-3',
        'cloud': 'maritaca',
        'base_url': 'https://chat.maritaca.ai/api'
    },

    {
        'name': 'sabia_3',
        'checkpoint': 'sabia-3',
        'cloud': 'maritaca',
        'base_url': 'https://chat.maritaca.ai/api',
    }

]

EMBEDDING_MODELS = {
    'name': 'multilingual_e5_large',
    'ef': 'LocalHuggingFaceEmbeddingFunction',
    'model': 'intfloat/multilingual-e5-large'
}


# Retorna modelos da HuggingFace
def get_huggingface_llm(llm_model):

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True
    )

    tokenizer = AutoTokenizer.from_pretrained(llm_model, token=HF_API_KEY)

    model_config = AutoConfig.from_pretrained(
        llm_model,
        trust_remote_code=True,
        max_new_tokens=256,
    )

    model = AutoModelForCausalLM.from_pretrained(
        llm_model,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
    )

    pipe = pipeline(
        task='text-generation',
        model=model,
        tokenizer=tokenizer,
        # max_new_tokens=256,
        # pad_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # print("Memory: ", model.get_memory_footprint())

    return llm


# Prompt Template

with open(file=directories.PROMPT_FILE_PATH, mode='r') as prompt_file:
    PROMPT_TEMPLATE = prompt_file.read()


def realizar_pergunta(query: str, llm_model, llm_model_name, relevant_context) -> str:

    # Load retrieved context and user query in the prompt template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # Call LLM model to generate the answer based on the given context and query
    prompt = prompt_template.format(contexto=relevant_context, pergunta=query)

    response_text = llm_model.invoke(prompt)

    if llm_model_name == 'maritaca':
        return response_text.content

    return response_text.split('RESPOSTA:')[-1]


def if_not_exist_create_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)  # aqui criamos a pasta caso nao exista


# Convert and write JSON object to file
def save_json(path, collection_dict):
    with open(file=path, mode='w') as outfile:
        json.dump(collection_dict, outfile, ensure_ascii=False, indent=4)


def do_query_to_llm(lenghtc_list: list, experiments: list, n_samples: int,
                    top_k_retrieval: int = 5):

    model_embedding_function = get_embedding_function()

    corpus = get_manual_content.load_questions()[0]

    if n_samples > 0:
        corpus = corpus[:n_samples]

    for llm_model in LLM_MODELS:

        if llm_model['cloud'] == 'localhost':
            model = get_huggingface_llm(llm_model['model'])
        else:
            model = ChatOpenAI(
                model=llm_model['checkpoint'],
                temperature=0.0,
                api_key=SABIA_API_KEY,
                base_url=llm_model['base_url'],
                max_tokens=256
            )

        chunk_files = tools.load_name_list_json(directories.CHUNKS_PATH)

        for chunk_file in chunk_files:

            chunk_name = chunk_file.split('/')[-1].split('.')[0]

            chunk_size = chunk_name.split('_')[-1]

            if chunk_size.isdigit():
                chunk_size = int(chunk_size)

            if chunk_size == 8000:
                top_k_retrieval = 4
            else:
                top_k_retrieval = 5

            if chunk_size in lenghtc_list:

                collection_ = collections_.get_db_embedding(model_embedding_function, chunk_name)

                for experiment in experiments:

                    os.makedirs(directories.ANSWERS_PATH + chunk_name.split('.')[0],
                                exist_ok=True)

                    path_name_json = os.path.join(directories.ANSWERS_PATH,
                                                  chunk_name.split('.')[0],
                                                  f'{llm_model["name"]}_{experiment}.json')

                    list_ids = []
                    list_sources = []
                    list_questions = []
                    list_ground_truth_answers = []
                    list_ground_truth_contexts = []
                    list_retrieved_contexts = []
                    list_llm_answers = []
                    list_modified_questions = []

                    print()

                    set_ids = set()

                    if os.path.exists(path_name_json):

                        with open(file=path_name_json, mode='r', encoding='utf-8') as json_file:

                            outputs = json.load(json_file)

                            list_ids = outputs['ids']

                            set_ids = set(list_ids)

                            if 'sources' in outputs:
                                list_sources = outputs['sources']

                            list_questions = outputs['questions']

                            list_ground_truth_answers = outputs['ground_truth_answers']

                            list_ground_truth_contexts = outputs['ground_truth_contexts']

                            list_retrieved_contexts = outputs['retrieved_contexts']

                            list_llm_answers = outputs['llm_answers']

                            list_modified_questions = outputs['modified_questions']

                    description = (f'\t\tGenerating Answers from {chunk_name}_{experiment} '
                                   f'with {llm_model["name"]}')

                    with tqdm(total=len(corpus), file=sys.stdout, colour='blue',
                              desc=description) as pbar:

                        for entry in corpus:

                            if entry['id'] in set_ids:

                                pbar.update(1)

                                continue

                            query = entry['pergunta']

                            list_questions.append(query)

                            if experiment == 'multiple_queries':

                                queries = multiple_queries.get_questions(model, query)

                                list_modified_questions.append(queries)

                                queries.append(query)

                                list_relevant_context = []

                                for query in queries:

                                    retrieved_context = collections_.do_query_to_db_embeddings(
                                        db_embedding=collection_,
                                        query=query,
                                        top_k_retrieval=2
                                    )

                                    list_relevant_context.append(retrieved_context)

                                relevant_context = '\n'.join(list_relevant_context)

                            else:

                                if experiment == 'rewrite':

                                    query = rewriter_question.get_reformulated_question(
                                        model, query
                                    )

                                    list_modified_questions.append(query)

                                elif experiment == 'hyde':

                                    query = hyde.get_answer(model, query)

                                    list_modified_questions.append(query)

                                relevant_context = collections_.do_query_to_db_embeddings(
                                    db_embedding=collection_,
                                    query=query,
                                    top_k_retrieval=top_k_retrieval
                                )

                            answer = realizar_pergunta(query, model, llm_model['cloud'],
                                                       relevant_context)

                            list_ids.append(entry['id'])
                            list_sources.append(entry['fonte'])
                            list_ground_truth_answers.append(entry['resposta'])
                            list_ground_truth_contexts.append(entry['contexto'])

                            list_retrieved_contexts.append(relevant_context)

                            list_llm_answers.append(answer)

                            pbar.update(1)

                            del relevant_context

                            result = {
                                'llms_model': llm_model['name'],
                                'ids': list_ids,
                                'questions': list_questions,
                                'sources': list_sources,
                                'modified_questions': list_modified_questions,
                                'ground_truth_answers': list_ground_truth_answers,
                                'ground_truth_contexts': list_ground_truth_contexts,
                                'llm_answers': list_llm_answers,
                                'retrieved_contexts': list_retrieved_contexts,
                            }

                            tools.save_json(path_name_json, result)

                del collection_

                # Pausa a execução por 1 segundos
                time.sleep(1)

        del model

        # Pausa a execução por 1 segundos
        time.sleep(1)
