def get_reasoning_prompt(class_1, class_2, domain, question, type, ):
    if type in ['推理', '应用']:
        prompt = "以下是一道{}{}题目，请你认真读题，一步一步思考并得出结论，并将过程严格按照如下格式输出:\n[思考过程]：根据题目可以推测，... \n[结论]：答案是 ... 。\n题目如下：{}".format(domain, type, question)
        # prompt = "以下是一道{}{}题目，请你认真读题，一步一步思考并得出结论，并将过程严格按照如下格式输出:\n<思考>... <思考完毕>\n<结论> ... <end>\n题目如下：{}".format(domain, type, question)
    if type == '证明':
        # prompt = "以下是一道{}{}题目，请你认真读题，一步一步思考，并将证明过程严格按照如下格式输出:\n<证明>... <证明完毕>\n题目如下：{}".format(domain, type, question)
        prompt = "以下是一道{}{}题目，请你认真读题，一步一步思考，并将证明过程严格按照如下格式输出:\n[证明过程]：根据题目可以得出，... \n题目如下：{}".format(domain, type, question)

    return prompt


