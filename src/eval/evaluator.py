"""
LongBench v1 official evaluation functions and prompt templates.
Source: https://github.com/THUDM/LongBench
"""
import re
import string
from collections import Counter
from rouge import Rouge
from fuzzywuzzy import fuzz

# Official per-task max_new_tokens
TASK_MAX_NEW = {
    "narrativeqa":        128,
    "qasper":             128,
    "multifieldqa_en":     64,
    "hotpotqa":            32,
    "2wikimqa":            32,
    "musique":             32,
    "gov_report":         512,
    "qmsum":              512,
    "multi_news":         512,
    "trec":                64,
    "triviaqa":            32,
    "samsum":             128,
    "passage_count":       32,
    "passage_retrieval_en": 32,
    "lcc":                 64,
    "repobench-p":         64,
}

# Official per-task prompt templates
DATASET2PROMPT = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news.\n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}

# Official per-task metric mapping
DATASET2METRIC = {
    "narrativeqa":         "qa_f1",
    "qasper":              "qa_f1",
    "multifieldqa_en":     "qa_f1",
    "hotpotqa":            "qa_f1",
    "2wikimqa":            "qa_f1",
    "musique":             "qa_f1",
    "gov_report":          "rouge",
    "qmsum":               "rouge",
    "multi_news":          "rouge",
    "trec":                "classification",
    "triviaqa":            "qa_f1",
    "samsum":              "rouge",
    "passage_count":       "count",
    "passage_retrieval_en":"retrieval",
    "lcc":                 "code_sim",
    "repobench-p":         "code_sim",
}

# Official metric functions
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall    = 1.0 * num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)

def qa_f1_score(prediction, ground_truth, **kwargs):
    pred_tokens  = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    return f1_score(pred_tokens, truth_tokens)

def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]

def code_sim_score(prediction, ground_truth, **kwargs):
    all_lines = prediction.lstrip('\n').split('\n')
    prediction = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            prediction = line
            break
    return fuzz.ratio(prediction, ground_truth) / 100

def classification_score(prediction, ground_truth, **kwargs):
    all_classes = kwargs.get("all_classes", [])
    em_match_list = []
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0

def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth))
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)

def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r'Paragraph (\d+)'
    matches = re.findall(pattern, ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth_id))
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)

# Unified scoring
def score_sample(prediction: str, answers: list, task: str, all_classes=None) -> float:
    if not answers:
        return 0.0
    if isinstance(answers, str):
        answers = [answers]

    metric = DATASET2METRIC.get(task, "qa_f1")

    if metric == "qa_f1":
        return max(qa_f1_score(prediction, a) for a in answers)
    elif metric == "rouge":
        return max(rouge_score(prediction, a) for a in answers)
    elif metric == "classification":
        return max(
            classification_score(prediction, a, all_classes=all_classes or [])
            for a in answers
        )
    elif metric == "code_sim":
        return max(code_sim_score(prediction, a) for a in answers)
    elif metric == "count":
        return max(count_score(prediction, a) for a in answers)
    elif metric == "retrieval":
        return max(retrieval_score(prediction, a) for a in answers)
    else:
        return max(qa_f1_score(prediction, a) for a in answers)

# Prompt builder with LongBench-style middle truncation:
# the full prompt (instruction + context + question) is built first; if it is
# too long, the middle is removed and the head/tail are kept, so the question
# and the "Answer:" prompt are never truncated.
def build_prompt(sample: dict, task: str, tokenizer, max_length: int) -> str:
    template = DATASET2PROMPT.get(
        task,
        "Context: {context}\n\nQuestion: {input}\n\nAnswer:"
    )
    try:
        prompt = template.format(**sample)
    except KeyError:
        context  = sample.get("context", "")
        question = sample.get("input", "")
        prompt   = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

    tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized) > max_length:
        half = max_length // 2
        prompt = (tokenizer.decode(tokenized[:half], skip_special_tokens=True)
                  + tokenizer.decode(tokenized[-half:], skip_special_tokens=True))
    return prompt