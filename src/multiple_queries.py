from langchain_core.prompts import ChatPromptTemplate

from typing import List
from langchain_core.output_parsers import BaseOutputParser


class LineListOutputParser(BaseOutputParser[List[str]]):
    """Parse out a question from each output line."""

    def parse(self, text: str) -> List[str]:
        lines = text.split('Muntiplas_Perguntas:')[-1].strip().split('\n')
        lines = list(set(lines))
        lines =  list(filter(None, lines))
        return lines[0:4]

questions_parser = LineListOutputParser()


def get_questions(llm, user_question):
    rewriter_prompt_template = """Você é um assistente virtual. 
    Sua tarefa é gerar de 1 a 4 versões diferentes da pergunta do usuário com objetivo de fornecer outras perspectivas.
    Forneça perguntas alternativas separadas por quebras de linha.             
    Retorne somente as perguntas alternativas, sem explicações, comentários ou qualquer outro texto adicional.

    Pergunta do Usuário: {user_question}
    Muntiplas_Perguntas:"""

    multi_query_gen_prompt = ChatPromptTemplate.from_template(rewriter_prompt_template)

    rewriter_chain = multi_query_gen_prompt | llm | questions_parser

    search_query = rewriter_chain.invoke(
        {
            'user_question': user_question
        }
    )

    return search_query
