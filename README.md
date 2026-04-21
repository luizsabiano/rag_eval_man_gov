Este projeto está integrado a Dissertação apresentada ao Programa de Pós-Graduação em Computação Aplicada do
Instituto Federal do Espírito Santo, como requisito parcial para obtenção do título de Mestre em
Computação Aplicada.

Autores: Luiz Sabiano F. Medeiros

Orientador: Prof. Dr. Hilário Tomaz Alves de Oliveira


# <p style="text-align:center;"> SISTEMA DESENVOLVIDO PARA AVALIAÇÃO DE ARQUITETURAS DE GERAÇÃO AUMENTADA POR RECUPERAÇÃO NO DOMÍNIO DE MANUAIS NORMATIVOS GOVERNAMENTAIS EM PORTUGUÊS DO BRASIL</p> 

---
## Resumo

<p style="text-align:justify;">

Devido à necessidade de padronização de condutas e procedimentos, organizações públicas comumente 
consolidam seus conhecimentos em manuais normativos. Entretanto, a vasta quantidade de informações 
contidas nestes documentos não garante o acesso eficaz à informação desejada. A ausência de sistemas 
de pesquisa eficientes muitas vezes resulta em desperdício de tempo, risco de execução de tarefas 
baseadas em diretrizes obsoletas e o comprometimento da qualidade do serviço público. 

Para solucionar a ineficiência na busca de informações relevantes, este trabalho teve como objetivo 
o desenvolvimento e a avaliação de um sistema de perguntas e respostas, no âmbito da técnica RAG, 
voltados a manuais normativos governamentais, em português. 

A metodologia proposta foi segmentada em duas fases com base nos componentes de recuperação e geração. 
Na **fase de recuperação**, foram avaliados cinco modelos de _embeddings_ de código aberto e dois proprietários
utilizando os conjuntos de dados públicos Pirá, FairyTaleQA PT-BR e SQuAD2 PT-BR, além de determinar 
$top$-$K$ que apresentou o melhor equilíbrio entre o custo computacional e a relevância do contexto 
recuperado. A **fase de geração** foi conduzida em dois estágios. O primeiro, baseado em dados públicos, 
avaliou oito modelos de linguagem de larga escala. O segundo estágio utilizou um conjuto de dados sintéticos 
desenvolvido para este trabalho, composto por 944 pares de perguntas e respostas derivados de manuais 
normativos governamentais. Usando esta base de dados, realizaram-se experimentos para definir a eficácia
de diversas estratégias de fragmentação. Posteriormente, analisou-se o impacto das técnicas de 
transformação de consulta (geração de documentos hipotéticos, reescrita da consulta e geração de 
múltiplas consultas) na qualidade das respostas geradas. 

Para a avaliação,  neste cenário governamental, adotou-se as métricas da família ROUGE e BERTScore, 
além das métricas de fidelidade e relevância das respostas. Os resultados dos experimentos apontaram 
que o Gemma 2 9B liderou em precisão e fidelidade nas bases públicas, superando modelos de maior escala
paramétrica. Em contrapartida, a família Sabiá 3 destacou-se em relevância e recall, evidenciando 
superioridade narrativa e argumentativa. No domínio governamental, a estratégia de fragmentação de 
500 tokens consolidou-se como a mais eficaz entre todas as métricas e arquiteturas avaliadas. 
As técnicas de transformação de consulta apresentaram desempenho inferior à abordagem padrão quando 
aplicadas em modo zero-shot. Na avaliação de fidelidade e relevância da resposta, os modelos da família 
Sabia 3, desmonstraram grande superioridade. 

</p>


## ESTRUTURAÇÃO DO TRABALHO

Visando reduzir a complexidade, o projeto foi estruturado 4 partes, acessíveis pelos links abaixo. 


### Conjunto de dados de governamental

[Gerador do dataset de manuais normativos governamentais](https://github.com/luizsabiano/manuals_question_generator) 

### Base de dados pública (Pirá, FairytaleQA PT-BR e SQuaD PT-BR)

[Avaliação dos modelos de recuperação e  geração](https://github.com/luizsabiano/rag_eval_ptbr) 

### Base de dados Governametal
[Extração e fragmentação dos documentos governamentais](https://github.com/luizsabiano/text_extractor_man_gov)

**Este projeto**: Avaliação dos modelos de geração e técnicas avançadas de recuperação   

 
## Estrutura de diretórios


<pre>├── <font color="#12488B"><b>data</b></font>
│   ├── <font color="#12488B"><b>answer</b></font> <font color="#7B68EE"> => Armazena as respostas geradas</font>
│   ├── <font color="#12488B"><b>chromadb</b></font> <font color="#7B68EE"> => Armazena os fragmentos de texto dos manuais vetorizados pelo chromadb</font>
│   ├── <font color="#12488B"><b>questions</b></font> <font color="#7B68EE"> => Armazena os conjunto de dados governamental</font>
│   ├── <font color="#12488B"><b>resources</b></font> <font color="#7B68EE"> => Armazena fontes. Ex. prompt</font>
│   ├── <font color="#12488B"><b>results</b></font> <font color="#7B68EE"> => Armazena o resultado da avaliação</font>
├── <font color="#12488B"><b>src</b></font>
│   ├── collections.py <font color="#7B68EE"> => Funções para persistência e consulta no chromadb</font>
│   ├── directories.py <font color="#7B68EE"> => Arquivo de caminhos (PATHs)</font>
│   ├── embedding_model.py <font color="#7B68EE"> => Configura modelo de embeddings</font>
│   ├── evaluation_measures.py <font color="#7B68EE"> => Avaliação, com RAGAS, do sistema</font>
│   ├── get_manual_content.py <font color="#7B68EE"> => Leitura das questões</font>
│   ├── hyde.py <font color="#7B68EE"> => Técnica Hyde </font>
│   ├── llm.py <font color="#7B68EE"> => Funções LLM </font>
│   ├── multple_queries.py <font color="#7B68EE"> => Funções para técnica multiplas consutas </font>
│   ├── rewriter_questions.py <font color="#7B68EE"> => Técnica RRR </font>
│   ├── similarity_measures.py <font color="#7B68EE"> => Avaliação por similaridade </font>
│   ├── summarize_rag_results.py <font color="#7B68EE"> => Sintetiza os resultados </font>
│   ├── tools.py  <font color="#7B68EE"> => Funções de apoio</font>
├── README.md
└── requirements.txt <font color="#7B68EE">O arquivo de requisitos para reproduzir os experimentos</font>
</pre>


### Instalação das dependências: 

$ pip install -r requirements.txt



## Execução 
Antes da execução crie um arquivo .env com as variáveis  a seguir, 
de acordo com o modelo que deseja utilizar.

HF_TOKEN="sua chave do Hugging Face"

OPENAI_API_KEY="sua chave do OpenAI"

SAMBANOVA_API_KEY="sua chave do SambaNova"

SABIA_API_KEY="sua chave do MaritacaAI"

Execute o arquivo main.py.

## Configuração 

1) list_splitting_parameters: Lista de parâmetros que configura o tamanho dos fragmentos
2) experiments: Tipos de experimentos serem realizados. Opções:
   [
        'chunk_size',
        'rewrite',
        'hyde',
        'multiple_queries'
    ]
3)  top_k_retrieval: Quantidade de fragmentos de texto recuperados.
4) is_prepare_environment_: Se True, indexa e persiste os chunks no banco de dados vetorial
5) is_run_generation: Se True, recupera contexto e gera resposta as questões predeterminadas
6) is_run_metrics: Se True, sumariza as avaliações
7) is_run_similarity_metrics: Se True, Avalia respostas geradas com Rouge e BertScore
8) is_run_ragas_metrics: Se True, avlia respostas com framework RAGAS.

Necessário GPU Nvidia.
