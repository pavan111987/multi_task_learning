import os
import json
import numpy as np
import tensorflow as tf
from absl import logging
from tqdm import tqdm

from transformers import DistilBertTokenizer, DistilBertConfig, TFDistilBertModel

epochs = 3
batch_size = 32
loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-05, beta_1=0.9, beta_2=0.999, epsilon=1e-06, clipnorm = 1.0)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")

def custom_loss_logits(train_labels, logits):
    pad_token_label_id = -1
    active_loss = tf.reshape(train_labels, (-1,)) != pad_token_label_id
    active_logits = tf.boolean_mask(tf.reshape(logits, (-1, 2)), active_loss)
    active_labels = tf.boolean_mask(tf.reshape(train_labels, (-1,)), active_loss)
    cross_entropy = loss_fct(active_labels, active_logits)
    loss = tf.reduce_sum(cross_entropy) * (1.0 / batch_size)
    return loss


def model_arch_multitask():
    num_labels = 2
    bert = TFDistilBertModel.from_pretrained("distilbert-base-cased")
    dropout = tf.keras.layers.Dropout(0.4)
    answer_logits = tf.keras.layers.Dense(10, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), name="answer_logits", activation = "softmax")
    classifier = tf.keras.layers.Dense(2, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), name="seq_logits")
    input_ids = tf.keras.layers.Input(shape = (None,), dtype=tf.int32)
    attention_mask = tf.keras.layers.Input(shape = (None,), dtype=tf.int32)
    outputs = bert([input_ids, attention_mask])
    answer_output = answer_logits(outputs[0][:, 0, :])
    sequence_output = outputs[0]
    sequence_output = dropout(sequence_output)
    logits = classifier(sequence_output)
    model = tf.keras.models.Model(inputs = [input_ids, attention_mask], outputs = [logits, answer_output])
    model.compile(loss={'seq_logits': custom_loss_logits, 'answer_logits': "categorical_crossentropy"}, optimizer=optimizer, loss_weights = {"answer_logits": 1.0, "seq_logits": 1.0}, metrics=["accuracy"])
    return model


def model_arch_tok_classification():
    num_labels = 2
    max_len = 128
    bert = TFDistilBertModel.from_pretrained("distilbert-base-cased")
    dropout = tf.keras.layers.Dropout(0.4)
    classifier = tf.keras.layers.Dense(2, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), name="seq_logits")
    question_input_ids = tf.keras.layers.Input(shape = (max_len,), dtype=tf.int32)
    question_attention_mask = tf.keras.layers.Input(shape = (max_len,), dtype=tf.int32)
    question_output = bert([question_input_ids, question_attention_mask])
    question_output = question_output[0][:, 0, :]
    question_output = tf.keras.layers.RepeatVector(max_len)(question_output)
    context_input_ids = tf.keras.layers.Input(shape = (max_len,), dtype=tf.int32)
    context_attention_mask = tf.keras.layers.Input(shape = (max_len,), dtype=tf.int32)
    outputs = bert([context_input_ids, context_attention_mask])
    sequence_output = outputs[0]
    sequence_output = tf.keras.layers.concatenate([sequence_output, question_output], axis = -1)
    sequence_output = dropout(sequence_output)
    logits = classifier(sequence_output)
    model = tf.keras.models.Model(inputs = [question_input_ids, question_attention_mask, context_input_ids, context_attention_mask], outputs = logits)
    model.compile(loss=custom_loss_logits, optimizer=optimizer)
    return model


if __name__ == "__main__":
    model = model_arch_multitask()

    model.summary()

    x = tokenizer.encode_plus("my name is pavan", "i work for genesys")
    x['input_ids'] = np.array([x['input_ids']])
    x['attention_mask'] = np.array([x['attention_mask']])

    y = [np.array([[np.random.randint(0, 2) for _ in range(x['input_ids'].shape[-1])]]), np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])]

    model.fit(x = [np.array(x['input_ids']), np.array(x['attention_mask'])], y = {'seq_logits': y[0], 'answer_logits': y[1]}, epochs = 4)

    ###################################################################################

    model = model_arch_tok_classification()

    model.summary()

    ip = []
    x = tokenizer.encode_plus("my name is pavan")

    ip.append(np.array([x['input_ids']]))
    ip.append(np.array([x['attention_mask']]))
    x = tokenizer.encode_plus("my name is pavan")

    ip.append(np.array([x['input_ids']]))
    ip.append(np.array([x['attention_mask']]))

    y = np.array([[np.random.randint(0, 2) for _ in range(len(x['input_ids']))]])

    model.fit(x = ip, y = y, epochs = 4)
