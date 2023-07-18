import re

choices_mapping = {
    2: "A、B",
    3: "A、B、C",
    4: "A、B、C、D",
    5: "A、B、C、D、E"

}
def processing_choices(class_2, question):
    question = re.sub(r"(\n\(A\)|\na\)|\nA\))", "\nA.", question)
    question = re.sub(r"(\n\(B\)|\nb\)|\nB\))", "\nB.", question)
    question = re.sub(r"(\n\(C\)|\nc\)|\nC\))", "\nC.", question)
    question = re.sub(r"(\n\(D\)|\nd\)|\nD\))", "\nD.", question)
    question = re.sub(r"(\n\(E\)|\ne\)|\nE\))", "\nE.", question)

    if class_2 != '翻译':
        question = re.sub(r"\na", "\nA.", question)
        question = re.sub(r"\nb", "\nB.", question)
        question = re.sub(r"\nc", "\nC.", question)
        question = re.sub(r"\nd", "\nD.", question)
        question = re.sub(r"\ne", "\nE.", question)

    if 'E' in question:
        return 5, question
    if 'D' in question:
        return 4, question
    if 'C' in question:
        return 3, question
    if 'B' in question:
        return 2, question


def get_multi_choice_prompt(class_1, class_2, domain, question):
    choices_mapping = {
        2: "A、B",
        3: "A、B、C",
        4: "A、B、C、D",
        5: "A、B、C、D、E"

    }
    choices, question = processing_choices(class_2, question)
    # prompt_with_cot = "以下是一道{}选择题，请你一步一步思考并将思考过程写在<解析>后，当思考结束，写上解析完毕。然后从{}中选出正确的答案，并写在<答案>和<end>之间。完整的题目回答的格式如下：\n<解析> ... <解析完毕>\n<答案> ... <end>\n请你严格按照上述格式作答。\n题目如下：{}".format(
        # class_2, choices_mapping[choices], question)
    prompt_with_cot = "以下是一道{}选择题，请你一步一步思考并将[思考过程]写下来，当思考结束，从{}中选出正确的答案，并将其写在[答案]之后。\n注意，请你严格按照给定的格式作答,完整的题目回答的格式如下：\n[思考过程]： ... \n[答案]： ...  \n题目如下：{}".format(class_2, choices_mapping[choices], question)

    # prompt_no_cot = "以下是一道{}选择题，请你从{}中选出正确的答案，并写在<答案>和<end>之间。完整的题目回答的格式如下：\n<答案> ... <end>\n请你严格按照上述格式作答。\n题目如下：{}".format(
    #     class_2, choices_mapping[choices], question)
    prompt_no_cot = "以下是一道{}选择题，请你从{}中选出正确的答案，并将其写在[答案]之后。\n注意，请你严格按照给定的格式作答,完整的题目回答的格式如下：\n[答案]： ... \n题目如下：{}".format(
        class_2, choices_mapping[choices], question)

    if class_2 in ['百科', '翻译', '文学']:
        prompt = prompt_no_cot
    else:
        prompt = prompt_with_cot

    return prompt