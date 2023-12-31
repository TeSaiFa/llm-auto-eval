[English](README_en.md)
# 目录


* [特赛发LLM模型评测简介](#特赛发LLM模型测评简介)
  * [评测宗旨](#评测宗旨)
  * [评测维度](#评测维度)
  * [评测方式](#评测方式)
  * [进一步的工作](#进一步的工作)
* [特赛发LLM模型测评一期主要结果](#模型测评一期主要结果)
* [实验代码说明](#实验代码说明)
  * [评测指令构建](#评测指令构建)
  * [大模型推理](#大模型推理)
  * [主观题自动评测](#主观题自动评测第一期为chatgpt进行的自动评测)
  * [评测结果标准化计算](#评测结果标准化计算)
  


# 特赛发LLM模型测评简介


## 评测宗旨
目前大多数针对大模型评测集中在两方面，一个是传统的NLU任务（语义理解，情感分析等）或者传统的NLP指标，另一个是更加与人类对齐的大规模权威性考试，然而实际生活中，我们更加关注模型在中文领域的实用性，即中文领域的大模型是否具有帮助人类解决具体任务的能力，为此，我们设计了中文领域大模型评测，构建了一个能够衡量「**模型是否好用**」的评测数据集。
## 评测维度
我们在评测过程中更加关注模型在中文领域的实用性，具体分为三个部分：
- **解决实际问题**：评估模型的实用性时，首先要考察它在解决实际问题上的表现。模型是否能够产生准确、有用的结果？它能否解决用户的需求或提供有效的建议？
- **真实数据集**：评测时使用真实世界的数据集来测试模型的性能，而非理想化的数据。这可以确保模型在实际应用中的表现。
- **多样性和广泛性**：模型评测应该涉及多样的应用场景，以验证模型的适用性。模型需要具备泛化能力，不仅仅在特定场景下表现优秀。

所以我们将关注的模型能力划定为**基础能力**、**中文特性**、**专业能力**三大一级分类，并在每个一级分类下设置了更多的二级小分类，以此为标准建立了我们的评测数据集。\
更多数据集的**详细信息**请点击[特赛发LLM评测第一期数据集概要](dataset_public/特赛发LLM评测第一期数据集概要.csv)
## 评测方式
由于时间关系，第一期测评主要评测了5个比较有代表性的中文领域大模型，同时加入了ChatGPT同分支的text-davinci-003（图称ChatGPT）作为比较，采用的方式是客观题使用准确率，主观题的开放性问答采用大模型当评测员的方法。
## 进一步的工作
我们的测评目前仍有许多不足，但会在之后做一些持续工作来不断完善和优化评测的体系和结果的展现，包括但不限于：
1. 数据集进一步完善，同时容纳更多的中文领域大模型加入评测榜单
2. 加入人工测评来保证结果的准确性, 同时思考模型测评如何向人工测评对齐
3. 细化我们的结果分析，例如，更加客观地展现人类/模型测评员测评中的偏好带来的结果偏差问题。

  

# 特赛发LLM模型测评一期主要结果
主要结果如下，详细结果请参考[特赛发LLM第一轮评测结果](dataset_public/eval_output/特赛发LLM第一轮评测结果.pdf)
## 三大综合能力榜单
![img.png](pics/scoreboard_of_the_three_capability.png)
### 基础能力细分榜单
![img.png](pics/basic_capability_radar_chart.png)
![img.png](pics/basic_capability_scoreboard.png)
### 中文特性细分榜单
![img.png](pics/chinese_radar_chart.png)
![img.png](pics/chinese_scoreboard.png)
### 专业能力细分榜单
![img.png](pics/professional_capability_radar_chart.png)
![img.png](pics/professional_capability_scoreboard.png)
## 客观题主观题打分结果
### 客观题
![img.png](pics/objective_question_acc.png)
### 主观题
- **三维度ChatGPT排序标准化得分**  

![img.png](pics/score_of_rank_in_subjective_question.png)

- **三维度ChatGPT打分标准化得分**  

![img.png](pics/score_from_3_dimension_in_subjective_question.png)

# 实验代码说明

当前repo主要包括以下几个部分

## 评测指令构建
参考了Gaokao等benchmark的构建方法，将评测问题的指令类别大致分为判断题、选择题、开放问答题和推理题四种类型，根据类别适配不同的指令，并经过线下实验调优，选择输出结果较好的指令进行测评结果的输出。  
详情参考```src/prompt```。

## 大模型推理
目前支持llama|moss|glm|baichuan|gptq系模型的推理，推理基于HF框架，也可以支持[fastllm](https://github.com/ztxz16/fastllm)框架(针对chatglm系列)，后续会兼容更多高效推理框架。
- **使用方法**
```
python llm_inference.py --model_name chatglm /
 --model_path your_model_path /
 --model_type [llama|moss|glm|baichuan|gptq] /
 --data_path your_data_path /
 --device 0  /
 --temperature 0 --top_p 1 --top_k 100 --rep_penalty 1.1 /
 --use_fastllm (optional for chatglm only) 
```

## 主观题自动评测(第一期为ChatGPT进行的自动评测)
我们让ChatGPT从三个维度来评价主观题的答案：**有效性、可靠性、流畅性。** 
参考了Sunweiwei等人的工作[RankGPT](https://github.com/sunnweiwei/RankGPT)以及复旦大学的模型评测工作[llm-eval](https://github.com/llmeval/llmeval-1)，我们最终选择了以下两种大模型评测方式：RankingGPT 和 ScoringGPT。
- RankingGPT：根据上述三个维度，我们让ChatGPT针对一个问题中所有大模型的回答（包含人工设定的参考答案）进行排序，得出1-7名。
- ScoringGPT：根据上述三个维度，我们让ChatGPT对每个模型的主观题回答分别进行打分，每个维度得分为从1-3分，每个分数都定义了一个具体标准，同时让模型认为参考答案的分数为满分（3，3，3）。
详情参考 ```scr/eval_agent```，使用示例见```auto-eval-with-chatgpt-examples.ipynb```。

## 评测结果标准化计算
- 打分结果标准化：从一级分类/二级分类入手，每个一级分类/二级分类满分设置为100分，按照比例为每道题目赋分。对于客观题，答案正确即得全部分数，对于主观题，我们结合题目在不同的分类下三个维度（有效性、可靠性、流畅性）的重要性，对ScoringGPT的打分结果进行加权，最终总得分为两者之和。
- 排序结果标准化：排序结果对应的是主观题，我们对不同名次赋予不同的权重(第一名的得分到第七名依次递减)，同时用频率进行标准化，最终得到一个排序转化后的分数。
计算过程和排序转化公式请参考```metrics_calculation.ipynb```