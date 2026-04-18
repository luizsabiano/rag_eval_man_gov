import os
import sys
import chromadb

from tqdm import tqdm
from src import directories, tools
from src.embedding_model import get_embedding_function

collections_dir = directories.CHROMA_PATH


# Retorna a collection salva
def get_db_embedding(model_embedding_function, collection_name):
    path_collection = directories.CHROMA_PATH + collection_name
    client = chromadb.PersistentClient(path=path_collection)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={'hnsw:space': 'cosine'},
        embedding_function=model_embedding_function
    )
    return collection


def do_query_to_db_embeddings(db_embedding, query, top_k_retrieval: int = 5):
    docs_chroma = db_embedding.query(
        query_texts=[query],
        n_results=top_k_retrieval,
    )
    relevant_context = '\n\n'.join([doc for doc in docs_chroma['documents'][0]])
    return relevant_context


def create_collection(collection_name: str):

    os.makedirs(collections_dir, exist_ok=True)

    path_collection = collections_dir + collection_name

    client = chromadb.PersistentClient(path=path_collection)

    model_embedding_function = get_embedding_function()

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={'hnsw:space': 'cosine'},
        embedding_function=model_embedding_function
    )

    return collection


def upsert_documents(collection, content):
    ids, documents, metadatas = content
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )


def insert_chromadb_content(lenghtc_list):

    chromabd_content = tools.get_chroma_content()

    collections_name = []

    print('\n')

    with tqdm(total=len(lenghtc_list), file=sys.stdout, colour='blue',
              desc='\t\tSavings chunks in database vector') as pbar:

        for content in chromabd_content:

            chunk_size = content['file'].split('_')[-1]

            if chunk_size.isdigit():
                chunk_size = int(chunk_size)

            if chunk_size in lenghtc_list:
                collection = create_collection(content['file'])
                upsert_documents(collection, (content['id'], content['document'], content['metadata'] ))
                pbar.update(1)
                collections_name.append(content['file'])

    print('\n\tForam criadas as coleções a seguir:\n')

    for collecion_name in collections_name:

        print(f'\t  Collection: {collecion_name}')
