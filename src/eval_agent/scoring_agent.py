from .basic import LLMAgent

class ScoringAgent(LLMAgent):
    """
    scoring answers to subjective questions
    """
    def __init__(self):
        super().__init__()

    def get_prompt(self, **kwargs):
        question = kwargs.get('question')
        llm_answer = kwargs.get('llm_answer')
        reference_answer = kwargs.get('reference_answer')
        scoring_prompt = """
You are a scoring GPT, an intelligent assistant that can score the LLM output answer to the question with respect to following three aspects with score 1 to 3.

Aspect 1 effectiveness:
Score 1 means the answer is totally irrelevant to the question;
Score 2 means the answer can partially follow the question and provide insufficient information to the question;
Score 3 means the answer can follow the question well and provide enough information to the question.

Aspect 2 reliability:
Score 1 means the answer is totally fictitious and illogical;
Score 2 means the answer contains some facts faults and full of logical flaws;
Score 3 means the answer is logically sound and totally factual.

Aspect 3 fluency:
Score 1 means the answer is incomplete and incoherent, or keeps repeating the same phrase;
Score 2 means the answer is complete, while containing grammatical errors;
Score 3 means the answer is complete, coherent and free of grammatical errors.

Please output the score and the output format should be 
[effectiveness score] ...
[reliability score] ...
[fluency score] ...

For example:
[effectiveness score] 2
[reliability score] 2
[fluency score] 2

The reference answer to the question will be given below, and suppose it scores 2 points.

Question is: 
{}

The LLM Answer is: 
{}

The reference answer to user's question is:
{}

The score of the LLM Answer is:
""".format(question, llm_answer, reference_answer)
        return scoring_prompt
