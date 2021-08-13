"""
Title : 더미 데이터를 활용한 감정 분석 모델링

"""
import tensorflow
from tensorflow.keras import preprocessing, Sequential, layers, optimizers, Model

sample = ['너는 오늘 이뻐 보인다',
          '나는 오늘 기분이 더러워',
          '나는 좋은 일이 생겼어',
          '아 오늘 진짜 짜증나',
          '환상적인데 정말 좋은거 같아']

label = [[1], [0], [1], [0], [1]]

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sample)

sequnces = tokenizer.texts_to_sequences(sample)
word_index = tokenizer.word_index

#

batch_size = 2
num_epochs = 100
vocab_size = len(word_index) + 1
emb_size = 128
hidden_dimension = 256
output_dimension = 1


# model


class Emotion(Model):
    def __init__(self, vocab_size, emb_size, hidden_dimension, output_dimension):
        super(Emotion, self).__init__(name='Emotion_model')
        self.emb = layers.Embedding(vocab_size, emb_size, input_length=4)
        self.den1 = layers.Dense(hidden_dimension, activation='relu')
        self.den2 = layers.Dense(output_dimension, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        x = self.emb(inputs)
        x = tensorflow.reduce_mean(x, axis=1)
        x = self.den1(x)
        x = self.den2(x)
        return x


model = Emotion(vocab_size, emb_size, hidden_dimension, output_dimension)
model.compile(optimizer=optimizers.Adam(0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(sequnces, label, epochs=num_epochs, batch_size=batch_size)

for _ in range(10):
    get_input = input('감정:')
    test_text = tokenizer.texts_to_sequences([str(get_input)])
    print(test_text)
    print(model.predict(test_text))