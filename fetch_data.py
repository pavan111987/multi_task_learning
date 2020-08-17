from tensorflow import keras

import random
import json


train_data_url_v1 = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
train_path_v1 = keras.utils.get_file("train.json", train_data_url_v1)
eval_data_url_v1 = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
eval_path_v1 = keras.utils.get_file("eval.json", eval_data_url_v1)

train_data_url_v2 = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
train_path_v2 = keras.utils.get_file("train.json", train_data_url_v2)
eval_data_url_v2 = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
eval_path_v2 = keras.utils.get_file("eval.json", eval_data_url_v2)


with open(train_path_v1) as f:
    raw_train_data = json.load(f)
    _ = raw_train_data.pop('version')

with open(eval_path_v1) as f:
    raw_eval_data = json.load(f)
    _ = raw_eval_data.pop('version')


with open(train_path_v2) as f:
    raw_train_data['data'].extend(json.load(f)['data'])


with open(eval_path_v1) as f:
    raw_eval_data['data'].extend(json.load(f)['data'])


class DataSample:

    @staticmethod
    def _get_negative_answers(index, data, num_negative_answers):
        negative_answers = []
        indices = list(range(len(data)))
        _ = indices.pop(index)
        negative_answers = list(set([context['context'] for ans_idx in random.sample(indices, num_negative_answers) for context in data[ans_idx]['paragraphs']]))
        return random.sample(negative_answers, num_negative_answers)

    @staticmethod
    def _get_targets(context, answers):
        targets = ""
        for ans in answers:
            start_idx = ans['answer_start']
            end_idx = start_idx + len(ans['text'])
            ans_tokens = ['ANSWER' for _ in ans['text'].strip().split()]
            targets = " ".join(context[0: start_idx].strip().split() + ans_tokens + context[end_idx: ].strip().split())
        return ['NOT_ANSWER' if t != 'ANSWER' else t for t in targets.split()]

    def __init__(self, index, title, context, question, answers, data):
        self.num_negative_answers = 1
        self.index = index
        self.title = title.replace('_', ' ')
        self.context = context
        self.question = question
        self.targets = self._get_targets(context, answers)
        self.negative_answers = self._get_negative_answers(index, data, self.num_negative_answers)


class DataSamples:

    def __init__(self, data):
        self.data_samples = []
        for index, sample in enumerate(data):
            for paragraph in sample['paragraphs']:
                for qa in paragraph['qas']:
                    self.data_samples.append(DataSample(index, sample['title'], paragraph['context'], qa['question'], qa['answers'], data))


train_data_samples = DataSamples(data = raw_train_data['data']).data_samples
test_data_samples = DataSamples(data = raw_eval_data['data']).data_samples
