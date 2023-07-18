import re

class AnswerProcessor:
    def processing(self, typ, model_name, answer):

        answer = self.processing_generative_qa(model_name, answer)
        if typ == '判断':
            answer = self.processing_judgement_qa(model_name, answer)
        if typ == '选择':
            answer = self.processing_multi_choice_qa(model_name, answer)

        return answer

    def processing_generative_qa(self, model_name, answer):
        if model_name in ['chinese-alpaca-13b', 'baichuan-7b']:
            special_token = '</s>'
            answer = answer.strip(special_token)
        if model_name =='moss-sft':
            special_token_1 = '<eoh>'
            special_token_2 = '<eom>'
            answer = answer.split(special_token_1)[-1].strip(special_token_2)
            answer = re.sub('<\|MOSS\|>: ','', answer).strip().strip('\n')
        else:
            if isinstance(answer, str):
                answer = answer.strip()
        return answer

    def processing_judgement_qa(self, model_name, answer):
        if model_name == 'answer':
            if re.search(r"(对|正确)", answer):
                res = True
            if re.search(r"(错|错误|不正确|不对|不能)", answer):
                res = False

        else:
            res = answer
        return res

    def processing_multi_choice_qa(self, model_name, answer):
        if not isinstance(answer, str):
            return None
        if model_name == 'answer':
            answer = answer.upper()
        else:
            if re.search(r'(?<=\[答案\](：|:))[A-E]', answer):
                answer = re.search(r'(?<=\[答案\](：|:))[A-E]', answer)[0]
            elif len(re.findall(r'[A-E]', answer)) == 1:
                answer = re.findall(r'[A-E]', answer)[0]
            else:
                find_patterns = [
                    r'(?<=答案(为|是))(\s)*[A-E]',
                    r'(?<=是)(\s)*[A-E]',
                    r'(?<=选)(\s)*[A-E]',
                    r'(?<=选项)[A-E]',
                    r'(?<=选择)[A-E]',
                ]
                for pattern in find_patterns:
                    if re.search(pattern, answer):
                        answer = re.search(pattern, answer)[0]
                        break

        return answer