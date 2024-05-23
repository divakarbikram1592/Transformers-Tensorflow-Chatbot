import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer

from load_dataset import LoadDataset
from load_trained_model import LoadTrainedModel
from predict import Predict


class ChatApp:

    def __init__(self):
        self.tokenizer = None
        self.model = None

    # def create_padding_mask(self, x):
    #     mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    #     # (batch_size, 1, 1, sequence length)
    #     return mask[:, tf.newaxis, tf.newaxis, :]

    def load_data(self):
        questions, answers = LoadDataset().load_conversations()

        # Build tokenizer using tfds for both questions and answers
        # self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        #     questions + answers, target_vocab_size=2 ** 13
        # )

        self.tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            questions + answers, target_vocab_size=2 ** 13
        )

    def get_response(self, query=""):
        if self.tokenizer is None:
            print("data-1")
            self.load_data()
            self.get_response()
        elif self.model is None:
            print("data-2")
            self.model = LoadTrainedModel().get_trained_model()
            self.get_response()
        else:

            pred_instance = Predict(self.model, self.tokenizer)

            answer = pred_instance.predict(query)
            return answer




# instance = ChatApp()
# instance.get_response()


