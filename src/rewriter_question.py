from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def get_reformulated_question(llm, user_question):

    rewriter_prompt_template = """Você é um assistente virtual responsável por reformular perguntas.
    Reformule a pergunta do usuário para que seja mais concisa e direta. 
    A nova pergunta deve: 
        1) capturar a intenção principal do usuário e 
        2) usar apenas os termos-chave do texto fornecido.     
    Se a reformulação não for possível, retorne "NONE". 
    Retorne somente a pergunta reformulada, sem explicações, comentários ou qualquer outro texto adicional.   
                
    Pergunta do Usuário: {user_question}
    Pergunta reformulada:"""

    rewriter_prompt = ChatPromptTemplate.from_template(rewriter_prompt_template)

    rewriter_chain = rewriter_prompt | llm | StrOutputParser()

    search_query = rewriter_chain.invoke(
        {
            'user_question': user_question
        }
    )

    new_user_question = search_query.split('Pergunta reformulada:')[-1]

    new_user_question = new_user_question.split('?')[0] + '?'

    return new_user_question.replace('\n', '').strip()