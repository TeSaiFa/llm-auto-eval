from .basic import LLMAgent

class RankingAgent(LLMAgent):
    """
    ranking answers to subjective questions
    """
    def __init__(self, method='relevance'):
        super().__init__()
        self.method = method

    def get_prompt(self, **kwargs):
        question = kwargs.get('question')
        llm_answer = kwargs.get('llm_answer')
        reference_answer = kwargs.get('reference_answer')
        llm_answer.append(reference_answer)
        answer_list = llm_answer
        if self.method == 'relevance':
            rank_prompt = """
This is RankGPT, an intelligent assistant that can rank answers based on their relevancy to the question.
The following are {} answers, each indicated by number identifier []. I can rank them based
on their relevance to question:

""".format(len(answer_list))

            for i, answer in enumerate(answer_list):
                rank_prompt += '[{}] {}\n'.format(i + 1, answer)
            rank_prompt += 'The question is: {}'.format(question)

            rank_prompt += """

I will rank the {} answers above based on their relevance to the question. The answer will be listed in descending order using identifiers, and the most relevant answer should be listed first, and the output format should be [] > [] > etc, e.g., [1] > [2] > etc.
The ranking results of the {} answers (only identifiers) is:""".format(len(answer_list), len(answer_list))

        if self.method == 'dimension':
            rank_prompt = """
This is RankGPT, an intelligent assistant that can rank answers based on their effectiveness, reliability, and fluency to the question. Definition of three principles are given below:
The definition of effectiveness are:
1. The answer can follow the question well and be related to the question;
2. The answer can provide some information to meet the requirements of the question.
The definition of reliability are:
1. The information in the answer is consistent with objective reality;
2. The answer and its reasoning are logically sound.
The definition of fluency are:
1. The answer is complete;
2. The answer is coherent and free of grammatical errors.

"""
            for i, answer in enumerate(answer_list):
                rank_prompt += '[{}] {}\n'.format(i + 1, answer)
            rank_prompt += 'The question is: {}'.format(question)

            rank_prompt += """

I will rank the {} answers above based on their effectiveness, reliability, and fluency to the question. The answer will be listed in descending order using identifiers, and the best answer should be listed first, and the output format should be [] > [] > etc, e.g., [1] > [2] > etc.
The ranking results of the {} answers (only identifiers) is:""".format(len(answer_list), len(answer_list))

        return rank_prompt

