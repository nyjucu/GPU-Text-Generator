import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, BatchNormalization, ReLU, Bidirectional
from tensorflow.keras.models import load_model

if __name__ == '__main__':
    with open('Schoolgirl.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    text = text.replace('\n', ' ')
    sentences = text.split('. ')

    longest_sentence = max(sentences, key=len)

    longest_sentence = longest_sentence.split()
    print(f"Longest sentence: {len(longest_sentence)}")
    # Longest sentence: 119

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)

    encoded_sentences = tokenizer.texts_to_sequences(sentences)
    print(f"Encoded sentences: {encoded_sentences[:5]}")

    w_counts = tokenizer.word_counts
    w_index = tokenizer.word_index

    print(f"Number of unique words and total vocab size: {len(tokenizer.word_counts) + 1}")
    vocab_size = len(tokenizer.word_counts) + 1

    sample = ["hide-and-seek"]
    print(tokenizer.texts_to_sequences(sample))

    max_length = longest_sentence

    examples = []

    for i in encoded_sentences:
        if len(i) > 1:
            for j in range(2, len(i)):
                examples.append(i[:j])

    print(f"Examples: {examples[:5]}")

    max_padded_length = len(longest_sentence)
    sequences = pad_sequences(examples, maxlen=max_padded_length, padding='pre')

    print(f"Sequences: {sequences[:5]}")
    print(f"Sequences shape: {sequences.shape}")

    X = sequences[:,:-1]
    y = sequences[:,-1]

    print(f"X shape, y shape: {X.shape}, {y.shape}")

    y = to_categorical(y, num_classes=vocab_size)
    print(f"y shape: {y.shape}")

    seq_length = X.shape[1]

    train = input("Train model? [y/n]\n")

    if train == 'y':
        model = Sequential()
        model.add(Input(shape=(seq_length,)))

        model.add(Embedding(vocab_size, 50))

        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(ReLU())

        model.add(LSTM(100, recurrent_dropout=0.2, return_sequences=True))
        model.add(Bidirectional(LSTM(50, recurrent_dropout=0.2)))

        model.add(Dense(100, activation="relu"))
        model.add(Dense(vocab_size, activation="softmax"))

        print(model.summary())

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(X, y, batch_size=256, epochs=150)

        model.save('generator2.keras')

    model = load_model('generator2.keras')

    def generate_text(input_text, no_words):
        for _ in range(no_words):
            encoded = tokenizer.texts_to_sequences([input_text])
            encoded = pad_sequences(encoded, maxlen=max_padded_length, padding='pre')
            y_pred = np.argmax(model.predict(encoded), axis=-1)

            predicted_word = ""

            for word, index in tokenizer.word_index.items():
                if index == y_pred:
                    predicted_word = word
                    break

            input_text = input_text + ' ' + predicted_word

        return input_text

    input_text = input("Input text:\n")

    while input_text != 'CMD_EXIT':
        generated_text = generate_text(input_text, 25)
        print(f"Generated text: {generated_text}")

        input_text = input("Input text:\n")