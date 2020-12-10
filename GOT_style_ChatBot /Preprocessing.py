import re
import json
import tensorflow_datasets as tfds
import tensorflow as tf


class Preprocessor():
    def __init__(self, path):
        self.path = path
        self.input = []
        self.output = []

    def preprocess_sentence(self, sentence):
        sentence = sentence.lower().strip()
        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
        sentence = sentence.strip()
        # adding a start and an end token to the sentence
        return sentence

    def load_dialouges(self):
        with open(self.path, 'r', encoding="utf-8") as f:
            data = json.load(f)

        for key in list(data.keys()):
            for i in range(len(data[key]) - 1):
                self.input.append(self.preprocess_sentence(data[key][i].split(':')[1]))
                self.output.append(self.preprocess_sentence(data[key][i + 1].split(':')[1]))

        return self.input, self.output

    def tokenizer_builder(self):
        tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            self.input + self.output, target_vocab_size=2 ** 13)
        START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
        # Vocabulary size plus start and end token
        VOCAB_SIZE = tokenizer.vocab_size + 2

        return tokenizer, VOCAB_SIZE, START_TOKEN, END_TOKEN

    def tokenize_and_filter(self, MAX_LENGTH, tokenizer, START_TOKEN, END_TOKEN):

        tokenized_inputs, tokenized_outputs = [], []

        for (sentence1, sentence2) in zip(self.input, self.output):
            # tokenize sentence
            sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
            sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
            # check tokenized sentence max length
            if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
                tokenized_inputs.append(sentence1)
                tokenized_outputs.append(sentence2)

        # pad tokenized sentences
        tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
        tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

        return tokenized_inputs, tokenized_outputs

    def dataset_builder(self, questions, answers, batch_size, buffer_size):
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'inputs': questions,
                'dec_inputs': answers[:, :-1]
            },
            {
                'outputs': answers[:, 1:]
            },
        ))

        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

