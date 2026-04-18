from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def get_answer(llm, user_question):

    hyde_prompt_template = """Você é um assistente virtual responsável por responder às perguntas dos usuários. 
        Responda à PERGUNTA a seguir de forma concisa e precisa, usando a norma culta do português do Brasil.         
        Caso não seja possível reponder, retorne a flag NONE.                
        Retorne somente a resposta, sem explicações, comentários ou qualquer outro texto adicional.

        PERGUNTA: {user_question}
        RESPOSTA:"""

    hyde_prompt = ChatPromptTemplate.from_template(hyde_prompt_template)

    hyde_chain = hyde_prompt | llm | StrOutputParser()

    search_query = hyde_chain.invoke(
        {
            'user_question': user_question
        }
    )

    return search_query.split('RESPOSTA:')[-1].replace('\n', '').strip()
