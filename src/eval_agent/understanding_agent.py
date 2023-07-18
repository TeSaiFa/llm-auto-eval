from .basic import LLMAgent

class UnderstandingAgent(LLMAgent):
    """
    understanding LLM's answers to objective questions
    """
    def __init__(self):
        super().__init__()

    def get_prompt(self, **kwargs):
        question = kwargs.get('question')
        llm_answer = kwargs.get('llm_answer')
        typ = kwargs.get('typ')
        if typ == 'judgement':
            judgement_prompt = """
Following is a judgement question and the corresponding answer, please tell me the attitude of answer to the question.
Here are three options for you to choose:
A. The answer totally thinks the argument in the question is correct
B. The answer partially agree or disagree with the argument in the question
C. The answer is irrelevant with the argument in the question, or does not answer the question

Question:{}
Answer:{}
Please output A/B/C only after the symbol [Answer]. 
For example:
[Answer] A

The attitude of the answer is:
[Answer]
""".format(question, llm_answer)
            return judgement_prompt