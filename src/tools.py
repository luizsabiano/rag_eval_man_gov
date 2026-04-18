import os
import json

from os import getcwd
JSON_PATH = getcwd() + "/data/chunks/"


# Recebe duas listas e cria outra lista com os numeros comuns entre elas (intersecção)
# Retorna a quantidade de elementos desta lista (quantidade de elementos comuns)

def if_not_exist_create_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)  # aqui criamos a pasta caso nao exista


def load_name_list_json(JSON_PATH):
    collection_list = []
    directory = os.listdir(JSON_PATH)
    if len(directory) != 0:
        arquivos = os.listdir(JSON_PATH)

        for arquivo in arquivos:
            if not arquivo.startswith('.'):
                collection_list.append(JSON_PATH + arquivo)

        del arquivos

        return collection_list
    return None


def files_json_to_dict(path):
    if os.path.exists(path):
        # Load json file and convert to dict
        with open(file=path, mode='r') as arquivo:
            dict_list = json.load(arquivo)
            return dict_list
    else:
        return None


# pega o arquivo json salvo após pesquisa nas collectins
# e cria insight dentro de um cvs
def json_to_chroma_format(list_jsons):
    id_ = []
    context = []
    metadata = []
    idx = 0
    for json_data in list_jsons:
        for chunks in json_data['chunks']:
            metadata.append({'source': json_data['source']})
            id_.append(str(idx))
            context.append(chunks)
            idx = idx + 1
    return id_, context, metadata


def get_chroma_content():
    json_list_of_paths = load_name_list_json(JSON_PATH)
    chromabd_content = []
    for json_file_path in json_list_of_paths:
        name = json_file_path.split('/')[-1].split('.')[0]
        json_data = files_json_to_dict(json_file_path)
        id_, context, metadata = json_to_chroma_format(json_data)
        chromabd_content.append({'file': name, 'id': id_, 'document': context, 'metadata': metadata})
    return chromabd_content


# Convert and write JSON object to file
def save_json(path, collection_dict):
    with open(file=path, mode='w') as outfile:
        json.dump(collection_dict, outfile, ensure_ascii=False, indent=4)





