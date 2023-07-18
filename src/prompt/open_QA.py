def get_open_qa_prompt(class_1, class_2, domain, question, type, ):
    if class_2 == '谚语':
        prompt = "“{}”这句谚语是什么意思？请解释一下？ ".format(question)
    elif class_2 == '诗文写作':
        prompt = "请用诗歌的形式对给定的主题进行自由创作\n以下是主题：{} ".format(question)
    elif class_2 == '翻译':
        prompt = "现在你是一个多语言的翻译助手, 请完成下面的翻译任务：\n{} ".format(question)
    else:
        prompt = '请你回答一个开放性问题，问题如下：\n{} '.format(question)
    prompt += '现在，请开始回答：'
    return prompt