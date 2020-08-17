from fetch_data import train_data_samples, test_data_samples
from build_model import model_arch_tok_classification, tokenizer, batch_size

from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.activations import softmax
import numpy as np


max_len = 128 # avg length of contexts is around 119

model = model_arch_tok_classification()

print (model.summary())

label2idx = {'ANSWER': 0, 'NOT_ANSWER': 1}

def encode_text_labels(text, targets):
    label_ids = []
    for idx, token in enumerate(text.split()):
        token_feats = tokenizer.encode(token)[1:-1]
        label_ids.append(label2idx[targets[idx]])
        if len(token_feats) > 1:
            label_ids.extend([-1 for _ in token_feats[1:]])
    label_ids = [-1] + label_ids + [-1]
    return label_ids


def encode_text_pairs(text1, text2):
    text1_input_ids = []
    text1_attention_mask = []
    for idx, token in enumerate(text1.split()):
        token_feats = tokenizer.encode(token)[1:-1]
        text1_input_ids.extend(token_feats)
    text1_input_ids = [101] + text1_input_ids + [102]
    text1_attention_mask = [1 for _ in text1_input_ids]
    text2_input_ids = []
    text2_attention_mask = []
    for idx, token in enumerate(text2.split()):
        token_feats = tokenizer.encode(token)[1:-1]
        text2_input_ids.extend(token_feats)
    text2_input_ids = [101] + text2_input_ids + [102]
    text2_attention_mask = [1 for _ in text2_input_ids]
    return text1_input_ids, text1_attention_mask, text2_input_ids, text2_attention_mask


x1 = []
x2 = []
x3 = []
x4 = []
y = []
for train_data_sample in tqdm(train_data_samples):
    text = train_data_sample.context
    text1 = train_data_sample.question
    text2 = train_data_sample.context
    targets = train_data_sample.targets
    text1_input_ids, text1_attention_mask, text2_input_ids, text2_attention_mask = encode_text_pairs(text1, text2)
    label_ids = encode_text_labels(text, targets)
    assert len(text2_input_ids) == len(label_ids)
    x1.append(text1_input_ids)
    x2.append(text1_attention_mask)
    x3.append(text2_input_ids)
    x4.append(text2_attention_mask)
    y.append(label_ids)
    for neg_ans in train_data_sample.negative_answers:
        text = neg_ans
        text2 = neg_ans
        targets = ['NOT_ANSWER' for _ in neg_ans.split()]
        text1_input_ids, text1_attention_mask, text2_input_ids, text2_attention_mask = encode_text_pairs(text1, text2)
        label_ids = encode_text_labels(text, targets)
        assert len(text2_input_ids) == len(label_ids)
        x1.append(text1_input_ids)
        x2.append(text1_attention_mask)
        x3.append(text2_input_ids)
        x4.append(text2_attention_mask)
        y.append(label_ids)



x1 = pad_sequences(x1, maxlen = max_len, padding = 'post', value = 0)
x2 = pad_sequences(x2, maxlen = max_len, padding = 'post', value = 0)
x3 = pad_sequences(x3, maxlen = max_len, padding = 'post', value = 0)
x4 = pad_sequences(x4, maxlen = max_len, padding = 'post', value = 0)

y = pad_sequences(y, maxlen = max_len, padding = 'post', value = -1)


test_x1 = []
test_x2 = []
test_x3 = []
test_x4 = []
test_y = []
for test_data_sample in tqdm(test_data_samples):
    text = test_data_sample.context
    text1 = test_data_sample.question
    text2 = test_data_sample.context
    targets = test_data_sample.targets
    text1_input_ids, text1_attention_mask, text2_input_ids, text2_attention_mask = encode_text_pairs(text1, text2)
    label_ids = encode_text_labels(text, targets)
    assert len(text2_input_ids) == len(label_ids)
    test_x1.append(text1_input_ids)
    test_x2.append(text1_attention_mask)
    test_x3.append(text2_input_ids)
    test_x4.append(text2_attention_mask)
    test_y.append(label_ids)
    for neg_ans in test_data_sample.negative_answers:
        text = neg_ans
        text2 = neg_ans
        targets = ['NOT_ANSWER' for _ in neg_ans.split()]
        text1_input_ids, text1_attention_mask, text2_input_ids, text2_attention_mask = encode_text_pairs(text1, text2)
        label_ids = encode_text_labels(text, targets)
        assert len(text2_input_ids) == len(label_ids)
        test_x1.append(text1_input_ids)
        test_x2.append(text1_attention_mask)
        test_x3.append(text2_input_ids)
        test_x4.append(text2_attention_mask)
        test_y.append(label_ids)



test_x1 = pad_sequences(test_x1, maxlen = max_len, padding = 'post', value = 0)
test_x2 = pad_sequences(test_x2, maxlen = max_len, padding = 'post', value = 0)
test_x3 = pad_sequences(test_x3, maxlen = max_len, padding = 'post', value = 0)
test_x4 = pad_sequences(test_x4, maxlen = max_len, padding = 'post', value = 0)

test_y = pad_sequences(test_y, maxlen = max_len, padding = 'post', value = -1)

model.fit([x1, x2, x3, x4], y, epochs = 4, batch_size=batch_size, validation_data = ([test_x1, test_x2, test_x3, test_x4], test_y))




#################################################################################################################

model.layers[-1].activation = softmax

model.save_weights("model_binaries/tf_model.h5")

question = 'what is a genius score?'
context = "Einstein never took a modern IQ test, but it's believed that he had an IQ of 160, the same score as Hawking. Only 1 percent of those who sit the Mensa test achieve the maximum mark, and the average score is 100. A 'genius' test score is generally considered to be anything over 140"

def predict_answers(question, context):
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    text = context
    text1 = question
    text2 = context
    text1_input_ids, text1_attention_mask, text2_input_ids, text2_attention_mask = encode_text_pairs(text1, text2)
    targets = ['NOT_ANSWER' for _ in context.split()]
    label_ids = encode_text_labels(text, targets)
    x1.append(text1_input_ids)
    x2.append(text1_attention_mask)
    x3.append(text2_input_ids)
    x4.append(text2_attention_mask)
    x1 = pad_sequences(x1, maxlen = max_len, padding = 'post', value = 0)
    x2 = pad_sequences(x2, maxlen = max_len, padding = 'post', value = 0)
    x3 = pad_sequences(x3, maxlen = max_len, padding = 'post', value = 0)
    x4 = pad_sequences(x4, maxlen = max_len, padding = 'post', value = 0)
    label_ids = pad_sequences([label_ids], maxlen = max_len, padding = 'post', value = -1)[0]
    preds = model([x1, x2, x3, x4])[0].numpy()
    preds = [p[0] for idx, p in enumerate(preds) if label_ids[idx] != -1]
    return [idx for idx, pred in enumerate(preds) if np.argmax(pred) == 0], [p[0] for idx, p in enumerate(preds) if label_ids[idx] != -1]



print(predict_answers(question, context))
