from os import  listdir
from os.path import isfile
from src import directories
from src import tools


# Lista todas as pastas em 1 nível a partir do diretório base
def list_dir(path):
    list_dirs = []
    # retorna uma lista contendo os nomes dos arquivos dentro do diretório fornecido
    for c in listdir(path):
        if not isfile(path + c):
            list_dirs.append(path + '/' + c)
    if not list_dirs:
        list_dirs.append(path)
    return list_dirs


# Lista todos os arquivos no caminho indicado
def list_files(path):
    list_files_ = []
    list_dirs = list_dir(path)
    for dir_path in list_dirs:
        for c in listdir(dir_path):
            if isfile(dir_path + '/' + c):
                c_split = c.split('.')[0]
                list_files_.append((c_split, dir_path + '/' + c))
    return list_files_

def load_questions():
    question_files_list = list_files(directories.QUESTIONS_PATH)
    json_list = []
    for question_file in question_files_list:
        json_file = tools.files_json_to_dict(question_file[1])
        json_list.append(json_file)
    return json_list
