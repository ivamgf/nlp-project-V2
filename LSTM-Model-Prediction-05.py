# Algorithm 1 - LSTM Model training
# Dataset cleaning, pre-processing XML and create slots and embeddings
# RNN Bidirectional LSTM Layer
# Results in file and browser

# -------------------------------------------------------------------------------------------
# Imports
import os
import xml.etree.ElementTree as ET
import webbrowser

import gensim
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from gensim.models import Word2Vec
import tensorflow as tf
from keras.layers import Dense
import datetime
import hashlib
import PyPDF2
from nltk.corpus import stopwords

# -------------------------------------------------------------------------------------------
# Downloads
nltk.download('punkt')

# Initialize the Word2Vec model
word2vec_model = gensim.models.Word2Vec(vector_size=100, window=5, min_count=1, sg=0)

# Directory paths
input_dir = "inputs"
output_dir = "outputs"
input_dir_prediction = "prediction"
# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)
files = [filename for filename in os.listdir(input_dir_prediction) if filename.endswith('.pdf')]

output_html = ""

output_html += "<h3>Predição:</h3>"

slot_number = 1

# -------------------------------------------------------------------------------------------
# Labels

#Others
label_1 = "[O]"

#B-ReqTreatment
label_2 = "[B-ReqTreatment]"

# -------------------------------------------------------------------------------------------
# Functions

# Tokenize the sentences into words and create skipgram Word2Vec
def tokenize_sentence(sentence):
    tokens = word_tokenize(sentence)
    tokens = [token.lower() for token in tokens if token.lower() not in string.punctuation]

    # Create skipgram Word2Vec model for the sentence
    model = Word2Vec(sentences=[tokens], min_count=1, workers=2, sg=1, window=5)

    return model

# -------------------------------------------------------------------------------------------
# Create a list to hold dictionaries for each word
word_dicts = []

# Loop through files in input directory
for filename in files:
    # print(filename)
    if filename.endswith('.pdf'):

        # Open the PDF file in binary mode
        with open(os.path.join(input_dir_prediction, filename), 'rb') as f:
            # Read the PDF content
            pdf_reader = PyPDF2.PdfFileReader(f)

            # Iterate through PDF file pages and extract text
            text = ""
            for page_num in range(pdf_reader.getNumPages()):
                page = pdf_reader.getPage(page_num)
                text += page.extractText()

            # Tokenize the content into sentences
            sentences_prediction = sent_tokenize(text)

            filtered_sentence_prediction_list = []
            # Create a list to hold word embeddings

            # Iterate through the sentences and tokenize the words
            for sentence_prediction in sentences_prediction:
                word_embeddings = []
                words_prediction = word_tokenize(sentence_prediction, language='portuguese')
                tokenized_stopwords = [word for word in words_prediction if
                                       word.lower() not in stopwords.words('portuguese')
                                       and not any(char.isdigit() or char in string.punctuation for char in word)]

                # Create a dictionary for each word
                for word_index, word in enumerate(tokenized_stopwords):
                    word_dict = {
                        'word': word,
                        'word_index': word_index,
                        'embedding': None,  # Placeholder for embedding
                        'sentence': sentence_prediction
                    }
                    if word in word2vec_model.wv:
                        word_embedding = word2vec_model.wv[word]
                        single_value_embedding = np.mean(word_embedding)
                        word_dict['embedding'] = single_value_embedding
                        word_embeddings.append(word_dict)

                output_html += "<pre>"
                output_html += f"<p>Dict: {word_dict}"
                output_html += "</pre>"

                if word_embeddings:
                    sentence_embedding = np.array([word_dict['embedding'] for word_dict in word_embeddings])
                else:
                    # Handle the case when no valid word embeddings are found
                    sentence_embedding = np.zeros(100)  # You can adjust the default value as needed

                sentence_embedding = np.expand_dims(sentence_embedding, axis=0)
                output_html += f"<p>Sentence Embeddings: {sentence_embedding}"

                # Put the filtered words back together into sentences
                filtered_sentence_prediction = ' '.join(tokenized_stopwords)

                filtered_sentence_prediction_list.append(tokenized_stopwords)

# -------------------------------------------------------------------------------------------
# Loop through files in input directory
for file in os.listdir(input_dir):
    if file.endswith(".xml"):
        xml_file = os.path.join(input_dir, file)
        tree = ET.parse(xml_file)
        root = tree.getroot()

        sentences = []

        # Loop through sentences
        for sentence in root.iter('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence'):
            tokens = []
            for token in sentence.iter('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token'):
                tokens.append(token.text.strip())

            # Checks if the sentence contains the specific tags
            if sentence.find(".//webanno.custom.Judgmentsentity") is not None:
                annotated_word = sentence.find(
                    ".//webanno.custom.Judgmentsentity/de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token").text.strip()

                sentence_text = ' '.join(tokens)

                # Checks if the sentence has already been added
                if sentence_text not in sentences:
                    sentences.append(sentence_text)

                    # Tokenize the sentences
                    sentences_list = sent_tokenize(sentence_text)

                    # Prints the sentences and the annotated word
                    for sent_idx, sent in enumerate(sentences_list[:10]):  # Select up to 10 sentences
                        tokenized_sent = tokenize_sentence(sent)
                        annotated_index = tokenized_sent.wv.key_to_index.get(
                            annotated_word.lower(), -1)
                        context_start = max(0, annotated_index - 10)
                        context_end = min(annotated_index + 11, len(tokenized_sent.wv.key_to_index))
                        context_words = list(tokenized_sent.wv.key_to_index.keys())[context_start:context_end]

                        # Print the Instance and Value attributes
                        for element in root.iter("webanno.custom.Judgmentsentity"):
                            if (
                                    "sofa" in element.attrib and
                                    "begin" in element.attrib and
                                    "end" in element.attrib and
                                    "Instance" in element.attrib and
                                    "Value" in element.attrib
                            ):
                                sofa = element.attrib["sofa"]
                                begin = element.attrib["begin"]
                                end = element.attrib["end"]
                                instance = element.attrib["Instance"]
                                value = element.attrib["Value"]

                        context_text = ' '.join(context_words)
                        context_text = context_text.replace(annotated_word, f"[annotation]{annotated_word}[annotation]")

                        # --------------------------------------------------------------------
                        # Word Embeddings
                        for word in context_words:
                            word_embedding = tokenized_sent.wv[word].reshape((100, 1))
                            word_embeddings.append(word_embedding)

                        # --------------------------------------------------------------------
                        # Bidirectional LSTM model
                        input_size = word_embedding.shape[-1]
                        hidden_size = 64
                        num_classes = 20
                        sequence_length = 1

                        # Generate example data
                        num_samples = 1
                        # Reshape the input data
                        X = word_embedding.reshape((num_samples, 1, 100))
                        y = tf.random.uniform((num_samples, num_classes))

                        # Create Bidirectional LSTM model
                        lstm_model = tf.keras.Sequential()
                        lstm_model.add(Dense(units=32))
                        lstm_model.add(
                            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                                hidden_size, input_shape=(1, 120), dropout=0.1)))
                        lstm_model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

                        # Learning rate
                        learning_rate = 0.01
                        rho = 0.9

                        # Optimizer
                        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=rho)

                        # Compile the model
                        lstm_model.compile(
                            loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

                        # Define patience and EarlyStopping
                        patience = 10
                        early_stopping = tf.keras.callbacks.EarlyStopping(
                            patience=patience, restore_best_weights=True, monitor='val_loss')
                        # Train the model with EarlyStopping
                        lstm_model.fit(X, y, epochs=60, batch_size=32, callbacks=[early_stopping])

                        # Print LSTM model results
                        # Calculate the min and max values of sentence_embedding
                        min_value = sentence_embedding.min(0)
                        max_value = sentence_embedding.max(1)

                        # Normalize sentence_embedding using min-max normalization
                        normalized_sentence_embedding = (sentence_embedding - min_value) / (
                                    max_value - min_value + 1e-8)  # Adding small constant to avoid division by zero

                        # Reshape the normalized_sentence_embedding
                        Z = normalized_sentence_embedding.reshape((num_samples, 1, 100))
                        # Z = sentence_embedding.reshape((num_samples, 1, 100))
                        lstm_results_prediction = []
                        lstm_results = lstm_model.predict(Z)

                        output_html += "<pre>"
                        output_html += "<p>Bidirectional LSTM Model Results:</p>"

                        # Loop through lstm_results
                        for lstm_result in lstm_results:
                            lstm_results_prediction.append(lstm_result)

                            # Loop through filtered_sentence_prediction_list
                            for filtered_words, lstm_result in zip(filtered_sentence_prediction_list,
                                                                   lstm_results_prediction):
                                word_indices = []
                                for word in filtered_words:
                                    word_lower = word.lower()
                                    if word_lower in tokenized_sent.wv:
                                        word_index = tokenized_sent.wv.key_to_index[word_lower]
                                        word_indices.append(word_index)
                                    else:
                                        # Handle words not found in the vocabulary
                                        word_indices.append(-1)  # Use -1 as a placeholder for missing words

                                results_dict = dict(zip(word_indices, lstm_result))
                                predict_x = 0.5

                                for word, word_index in zip(filtered_words, word_indices):
                                    result = results_dict.get(word_index, 0.0)  # Default to 0.0 if word index not found
                                    if word == annotated_word:
                                        result = results_dict.get(word_index, 1.0)
                                    # output_html += f"<p>{word}"

                                output_html += f"<p>{result}"

                                # Assign labels based on the predict_x threshold
                                if result <= predict_x:
                                    output_html += f" - {label_1}"
                                else:
                                    output_html += f" - {label_2}"

                                output_html += "</p>"
                                output_html += "<p>"
                                output_html +="-------------------------------------------------------"
                                output_html += "</p>"

                            output_html += "</pre>"

                        slot_number += 1

# --------------------------------------------------------------------
# Generate unique hash for the file titles
current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
hash_value = hashlib.md5(current_datetime.encode()).hexdigest()

# Output files paths
output_file_txt = os.path.join(output_dir, f"output_LSTM_{current_datetime}_{hash_value}.txt")
output_file_html = os.path.join(output_dir, f"output_LSTM_{current_datetime}_{hash_value}.html")

# Save the result to the output TXT file
with open(output_file_txt, "w", encoding="utf-8") as f:
    f.write(output_html)

# Save the result to the HTML output file
with open(output_file_html, "w", encoding="utf-8") as f:
    f.write(output_html)

# Opens the HTML file in the browser
webbrowser.open(output_file_html)

print("Results saved in 'outputs' folder")