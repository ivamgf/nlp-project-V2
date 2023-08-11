# Algorithm 1 - LSTM Model training
# Dataset cleaning, pre-processing XML and create slots and embeddings
# RNN Bidirectional LSTM Layer
# Results in file and browser

# -------------------------------------------------------------------------------------------
# Imports
import os
import xml.etree.ElementTree as ET
import webbrowser
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from gensim.models import Word2Vec
import tensorflow as tf
from keras.layers import Dense
import datetime
import hashlib
from PyPDF2 import PdfReader

# -------------------------------------------------------------------------------------------
# Downloads
nltk.download('punkt')

# Directory paths
input_dir = "inputs"
output_dir = "outputs"
input_dir_prediction = "prediction"
# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

output_html = ""

output_html += "<h3>Arquivos encontrados no diretório:</h3>"

slot_number = 1

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
# Loop through files in input directory
for file in os.listdir(input_dir_prediction):
    if file.endswith(".pdf"):
        pdf_file = os.path.join(input_dir_prediction, file)
        pdf_reader = PdfReader(pdf_file)

        # Extrair texto do PDF
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()

        # Tokenizar o texto
        sentences_prediction = sent_tokenize(pdf_text)
        sent_tokens = [word_tokenize(sentence_token) for sentence_token in sentences_prediction]

        # Exibir tokens
        print(f"Tokens do arquivo {file}:")
        for i, sentence_tokens in enumerate(sent_tokens, start=1):
            print(f"Sentença {i}: {sentence_tokens}")

# -------------------------------------------------------------------------------------------
# Loop through files in input directory
for file in os.listdir(input_dir):
    if file.endswith(".xml"):
        output_html += f"<p>{file}</p>"
        xml_file = os.path.join(input_dir, file)
        tree = ET.parse(xml_file)
        root = tree.getroot()

        output_html += f"<h4>Conteúdo do arquivo {file}:</h4>"
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
                        output_html += f"<p>Sentença {slot_number}: {context_text}</p>"
                        output_html += f"<p>Annotated Word: {annotated_word}</p>"
                        output_html += f"<p>Instance: {instance}</p>"
                        output_html += f"<p>Value: {value}</p>"

                        # Print the token vector
                        output_html += f"<p>Slot de Tokens {slot_number}: {context_words}</p>"

                        # --------------------------------------------------------------------
                        # Word Embeddings
                        output_html += f"<p>Word Embeddings {slot_number}: </p>"
                        output_html += "<pre>"
                        for word in context_words:
                            word_embedding = tokenized_sent.wv[word].reshape((100, 1))
                            output_html += f"<p>{word}: {word_embedding}</p>"
                        output_html += "</pre>"

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
                        early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)

                        # Train the model with EarlyStopping
                        lstm_model.fit(X, y, epochs=60, batch_size=32, callbacks=[early_stopping])

                        # Print LSTM model results
                        output_html += "<p>Bidirectional LSTM Model Results:</p>"
                        lstm_results = lstm_model.predict(X)
                        output_html += "<pre>"
                        # Get the indices of the words in the Slot de Tokens
                        word_indices = [tokenized_sent.wv.key_to_index[word.lower()] for word in context_words]
                        # Create a dictionary mapping word indices to LSTM results
                        results_dict = dict(zip(word_indices, lstm_results[0]))
                        # Iterate over the words in the Slot de Tokens and print the corresponding LSTM result
                        for word in context_words:
                            word_index = tokenized_sent.wv.key_to_index[word.lower()]
                            result = results_dict.get(word_index, 0.0)  # Default to 0.0 if word index not found
                            if word == annotated_word:
                                result = results_dict.get(word_index, 0.0)
                            output_html += f"<p>{word}: {result}"
                            if word != annotated_word:
                                output_html += " - [O]"
                            if word == annotated_word:
                                output_html += " - [B-ReqTreatment]"
                            output_html += "</p>"

                            # Map the predicted value to labels
                            if result < 0.5:  # Defina um limiar adequado para o seu caso
                                label = "O"
                            else:
                                label = "B-ReqTreatment"

                            output_html += f"<p>{word}: {label}</p>"

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