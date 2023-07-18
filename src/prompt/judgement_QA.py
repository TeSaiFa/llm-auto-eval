
def get_judgement_prompt(class_2, question):
    if class_2 == '语义理解':
        # prompt = '{}\n请一步一步思考并从“正确”或者“错误”两个选项中给出最终的判断。完整的题目回答的格式如下：\n<思考> ... <思考完毕>\n<答案> ... <end>\n请你严格按照上述格式作答。'.format(question)
        prompt = '{}\n请一步一步思考并从“正确”或者“错误”两个选项中给出最终的判断。\n请开始回答：'.format(question)

    if class_2 == '百科':
        # prompt = '{}\n请利用你的自然知识，从“正确”或者“错误”两个选项中给出最终的判断。完整的题目回答的格式如下：\n<答案> ... <end>\n请你严格按照上述格式作答。'.format(question)
        prompt = '{}\n请利用你的自然知识，一步一步思考并从“正确”或者“错误”两个选项中给出最终的判断。\n请开始回答：'.format(question)

    if class_2 == '计算能力':
        # prompt = '{}\n请利用你的高等数学知识，从“正确”或者“错误”两个选项中给出最终的判断。完整的题目回答的格式如下：\n<答案> ... <end>\n请你严格按照上述格式作答。'.format(question)
        prompt = '{}\n请利用你的高等数学知识，一步一步思考并从“正确”或者“错误”两个选项中给出最终的判断。\n请开始回答：'.format(question)

    if class_2 == '代码':
        # prompt = '{}\n请利用你的python知识，一步一步思考并从“正确”或者“错误”两个选项中给出最终的判断。完整的题目回答的格式如下：\n<思考> ... <思考完毕>\n<答案> ... <end>\n请你严格按照上述格式作答。'.format(question)
        prompt = '{}\n请利用你的python知识，一步一步思考并从“正确”或者“错误”两个选项中给出最终的判断。\n请开始回答：'.format(question)

    return prompt