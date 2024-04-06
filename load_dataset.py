import tensorflow as tf
import os
import re

class LoadDataset:

    def __init__(self):
        print("Loading dataset...")
        # Maximum number of samples to preprocess
        self.MAX_SAMPLES = 50000

    def download_dataset(self):
        path_to_zip = tf.keras.utils.get_file(
            "cornell_movie_dialogs.zip",
            origin="http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip",
            extract=True,
        )

        path_to_dataset = os.path.join(
            os.path.dirname(path_to_zip), "cornell movie-dialogs corpus"
        )

        path_to_movie_lines = os.path.join(path_to_dataset, "movie_lines.txt")
        path_to_movie_conversations = os.path.join(path_to_dataset, "movie_conversations.txt")

        return path_to_movie_lines, path_to_movie_conversations

    def preprocess_sentence(self, sentence):
        sentence = sentence.lower().strip()
        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        # removing contractions
        sentence = re.sub(r"i'm", "i am", sentence)
        sentence = re.sub(r"he's", "he is", sentence)
        sentence = re.sub(r"she's", "she is", sentence)
        sentence = re.sub(r"it's", "it is", sentence)
        sentence = re.sub(r"that's", "that is", sentence)
        sentence = re.sub(r"what's", "that is", sentence)
        sentence = re.sub(r"where's", "where is", sentence)
        sentence = re.sub(r"how's", "how is", sentence)
        sentence = re.sub(r"\'ll", " will", sentence)
        sentence = re.sub(r"\'ve", " have", sentence)
        sentence = re.sub(r"\'re", " are", sentence)
        sentence = re.sub(r"\'d", " would", sentence)
        sentence = re.sub(r"\'re", " are", sentence)
        sentence = re.sub(r"won't", "will not", sentence)
        sentence = re.sub(r"can't", "cannot", sentence)
        sentence = re.sub(r"n't", " not", sentence)
        sentence = re.sub(r"n'", "ng", sentence)
        sentence = re.sub(r"'bout", "about", sentence)
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
        sentence = sentence.strip()
        return sentence

    def load_conversations(self):

        # retrieve data
        path_to_movie_lines, path_to_movie_conversations = self.download_dataset()

        # dictionary of line id to text
        id2line = {}
        with open(path_to_movie_lines, errors="ignore") as file:
            lines = file.readlines()
        for line in lines:
            parts = line.replace("\n", "").split(" +++$+++ ")
            id2line[parts[0]] = parts[4]

        inputs, outputs = [], []
        with open(path_to_movie_conversations, "r") as file:
            lines = file.readlines()
        for line in lines:
            parts = line.replace("\n", "").split(" +++$+++ ")
            # get conversation in a list of line ID
            conversation = [line[1:-1] for line in parts[3][1:-1].split(", ")]
            for i in range(len(conversation) - 1):
                inputs.append(self.preprocess_sentence(id2line[conversation[i]]))
                outputs.append(self.preprocess_sentence(id2line[conversation[i + 1]]))
                if len(inputs) >= self.MAX_SAMPLES:
                    return inputs, outputs
        return inputs, outputs

    # def

