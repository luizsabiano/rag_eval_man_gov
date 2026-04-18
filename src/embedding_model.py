import torch

from transformers import BitsAndBytesConfig
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

EMBEDDING_MODELS = {
        'ef': 'LocalHuggingFaceEmbeddingFunction',
        'name': 'multilingual_e5_large',
        # 'model': 'intfloat/multilingual-e5-large',
        'model': 'intfloat/multilingual-e5-base',
        'normalize_embeddings': False
    }


class LocalHuggingFaceEmbeddingFunction(EmbeddingFunction[Documents]):

    def __init__(self, model_name: str):

        torch.cuda.empty_cache()

        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type='nf4',
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     llm_int8_enable_fp32_cpu_offload=True
        # )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = SentenceTransformer(
            model_name,
            # model_kwargs={'quantization_config': bnb_config}
        )

        self.model.to(device)

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input).tolist()


def get_embedding_function():
    embed_documents = LocalHuggingFaceEmbeddingFunction(EMBEDDING_MODELS['model'])
    return embed_documents
