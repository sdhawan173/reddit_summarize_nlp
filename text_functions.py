import os
import time
import string
import subprocess
import shlex
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# import sklearn.model_selection as sklms
# import tensorflow as tf
import text_file_operations as tfo

try:
    nltk.data.find('tokenizers/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.add('n\'t')
STOP_WORDS.add('\'s')
STOP_WORDS.add('\'m')
STOP_WORDS.add('\'ll')
STOP_WORDS.add('\'ve')
STOP_WORDS.add('\'re')
STOP_WORDS.add('\'d')

PUNCTUATION = list(set(string.punctuation))
PUNCTUATION.append('â–')
PUNCTUATION.append('br')
PUNCTUATION.append('<br>')
PUNCTUATION.append('<\\br>')
PUNCTUATION.append('’')
PUNCTUATION.append('”')
PUNCTUATION.append('“')
PUNCTUATION.append('``')
PUNCTUATION.append('/')
PUNCTUATION.append('\'\'')
PUNCTUATION.append('\"')
PUNCTUATION.append('\n')
PUNCTUATION.append('..')
PUNCTUATION.append('...')
PWD = os.getcwd()

POST_DICT = {}  # {lemmatized word: original word (1st variation only)}
WORD_FREQ = {}  # {lemmatized word: lemm. word frequency}
COMMENT_LIST_DICT = {}  # {comment.id: list preprocessed comment sentences}


def split_words(sentence):
    """
    Splits each comment from list of comments into words
    :param sentence: sentence string (punctuation not necessary)
    :return: list of lists of comments split into words
    """
    processed_words = []
    text_split_word = nltk.word_tokenize(sentence)
    stemmer = nltk.PorterStemmer()
    for word in text_split_word:
        word_lower = word.lower()
        if word_lower not in PUNCTUATION and word_lower not in STOP_WORDS:
            lemmatized_word = stemmer.stem(word_lower)
            processed_words.append(lemmatized_word)
            if lemmatized_word not in POST_DICT.keys():
                POST_DICT.update({lemmatized_word: word_lower})
                WORD_FREQ.update({lemmatized_word: 1})
            elif lemmatized_word in POST_DICT.keys():
                WORD_FREQ[lemmatized_word] = WORD_FREQ.get(lemmatized_word) + 1
    return processed_words


def split_sentence(comment_body):
    """
    Splits comments into sentences
    :param comment_body: individual comment from reddit
    :return: comments split into sentences
    """
    # Make sure 'punkt' is installed
    split_comment = nltk.sent_tokenize(comment_body)
    return split_comment


def preprocess(comment):
    comment_body = comment.body
    split_comment = []
    comment_sentence_split = split_sentence(comment_body)
    for sentence in comment_sentence_split:
        sentence_split_words = split_words(sentence)
        split_comment.append(sentence_split_words)
    COMMENT_LIST_DICT.update({comment.id: split_comment})


def replace_multiwords(word_list):
    """
    replaces individual words in list with words that may be multiwords
    :param word_list: list of words
    :return: list of words with multiword combinations replaced by a single string
    """
    finder = BigramCollocationFinder.from_words(word_list)
    finder.apply_freq_filter(3)
    collocations = finder.nbest(BigramAssocMeasures.pmi, 5)
    for collocation_to_merge in collocations:
        merged_words = []
        i = 0
        while i < len(word_list):
            if i < len(word_list) - 1 and (word_list[i], word_list[i + 1]) == collocation_to_merge:
                merged_words.append(' '.join(collocation_to_merge))
                i += 2
            else:
                merged_words.append(word_list[i])
                i += 1
        word_list = merged_words.copy()
    return word_list


def exclusion_filter(word_list):
    exclusion_list = []
    for word in word_list:
        if word.lower() not in STOP_WORDS and word.lower() not in PUNCTUATION:
            exclusion_list.append(word.lower())

    punc_list = list(PUNCTUATION)
    for word in exclusion_list:
        new_items = []
        word_index = None
        remove_boolean = None
        for character in word:
            if character in punc_list:
                word_index = exclusion_list.index(word)
                remove_boolean = True
                punc_index = punc_list.index(character)
                new_items = exclusion_list[word_index].split(punc_list[punc_index])

        if remove_boolean:
            exclusion_list.remove(exclusion_list[word_index])
            insert_count = 0
            for item in new_items:
                if item != '':
                    exclusion_list.insert(word_index + insert_count, item)
                    insert_count += 1
    return exclusion_list


def lemmatization(text_split_sentence, exclusion=False, multiword=False):
    """
    lemmatizes each comment from list of comments into lemmatized words
    :param text_split_sentence: list of lists of comments split into sentences
    :param exclusion: Boolean to exclude STOP_WORDS and PUNCTUATION
    :param multiword: Boolean to activate replace_multiwords function
    :return: list of lists of comments split into lemmatized words
    """
    print('\nLemmatizing {} comments ... '.format(format(len(text_split_sentence))))
    if exclusion:
        print('Excluding stop words and punctuation ...')
    if multiword:
        print('Replacing multiwords ...')
    lemmatizer = WordNetLemmatizer()
    all_comments_lemmatized = []
    if not multiword:
        for comment in text_split_sentence:
            comment_lemmatized = []
            for sentence in comment:
                words = [word.lower() for word in word_tokenize(sentence)]
                if exclusion:
                    words = exclusion_filter(words.copy())
                for word in words:
                    comment_lemmatized.append(lemmatizer.lemmatize(word))
            all_comments_lemmatized.append(comment_lemmatized)
    elif multiword:
        for comment in text_split_sentence:
            word_list = []
            comment_lemmatized = []
            for sentence in comment:
                word_list += [word.lower() for word in word_tokenize(sentence)]
            word_list = exclusion_filter(word_list.copy())
            word_list = replace_multiwords(word_list.copy())
            for word in word_list:
                comment_lemmatized.append(lemmatizer.lemmatize(word))
            all_comments_lemmatized.append(comment_lemmatized)
    return all_comments_lemmatized


def stemming(text_split_word, exclusion=False, multiword=False):
    """
    stems each comment from list of comments into stemmed words
    :param text_split_word: list of lists of comments split into sentences
    :param exclusion: Boolean to exclude STOP_WORDS and PUNCTUATION
    :param multiword: Boolean to activate replace_multiwords function
    :return: list of comments split into stemmed words
    """
    print('\nStemming {} comments ... '.format(format(len(text_split_word))))
    if exclusion:
        print('Excluding stop words and punctuation ...')
    if multiword:
        print('Replacing multiwords ...')
    stemmer = nltk.PorterStemmer()
    all_comments_stemmed = []
    for word_list in text_split_word:
        stemmed_comment = [stemmer.stem(word) for word in word_list]
        if exclusion:
            stemmed_comment = exclusion_filter(stemmed_comment.copy())
        if multiword:
            stemmed_comment = replace_multiwords(stemmed_comment.copy())
        all_comments_stemmed.append(stemmed_comment)
    return all_comments_stemmed


def lda(all_comments, n):
    """

    :param all_comments:
    :param n:
    :return:
    """
    print('\nPerforming Latent Drichlet Allocation ({} topics)...'.format(n))
    lda_dict = corpora.Dictionary(all_comments)
    lda_corpus = []
    for comment in all_comments:
        lda_corpus.append(lda_dict.doc2bow(comment))
    lda_model = LdaModel(lda_corpus, n, id2word=lda_dict)
    for output in lda_model.print_topics():
        print(output)
    return lda_model


def word2vec_cbow(data, vector_size, window, min_count, sg=0, epochs=30, verbose=True):
    if verbose:
        print('Running word2vec with CBOW\n'
              '(vector_size={}, window={}, min_count={}) ...'.format(vector_size, window, min_count))
    start = time.time()
    # sg=0 is CBOW
    word2vec_model = Word2Vec(data, vector_size=vector_size, window=window, min_count=min_count, sg=sg, epochs=epochs)
    end = time.time()
    total = '{:.3f}'.format(end - start)
    if verbose:
        print('Time to train: {} seconds'.format(total))
    return word2vec_model


def load_glove_model():
    return KeyedVectors.load_word2vec_format(PWD + '/glove/vectors.txt', binary=False, no_header=True)


def model_eval(eval_list, similarity_size, word2vec_model=None, glove_model=None):
    if word2vec_model is not None:
        print('Evaluating word2vec model ...')
    if glove_model is not None:
        print('Evaluating glove model ...')
    for word in eval_list:
        similar_words = ''
        if word2vec_model is not None:
            similar_words = word2vec_model.wv.most_similar(word.lower(), topn=similarity_size)
        if glove_model is not None:
            similar_words = glove_model.most_similar(word.lower(), topn=similarity_size)
        similar_words_output = [
            (output_word, '{:.2f}'.format(similarity * 100))
            for output_word, similarity in similar_words
        ]
        print('Words similar to \'{}\':{}'.format(word, similar_words_output))


def no_space_path(path):
    if path.__contains__(' '):
        path_components = path.split('/')
        path_components_escaped = [
            shlex.quote(component)
            if ' ' in component
            else component
            for component in path_components
        ]
        path = '/'.join(path_components_escaped)
        return path


def glove_world(data, shell_file='demo.sh', text_file_name='all_comments'):
    print('Glove main function ...')
    glove_pwd = os.getcwd() + '/' + 'glove'
    os.chdir(glove_pwd)

    print('Saving data for glove to use ...')
    tfo.save_data(data, PWD + '/glove/' + text_file_name + '.txt')

    print('Running \'./{}\'\n'
          '(corpus=all_comments.txt, vector_size=100, window=5, min_count=1) ...'.format(shell_file))
    # chmod +x demo.sh
    # chmod +x ... path ... /glove/*  spaces in path must be preceded with escape character \
    shell_pwd = os.path.join(PWD, 'glove/')
    print(shell_pwd)
    os.chdir(shell_pwd)
    shell_pwd = no_space_path(shell_pwd)
    shell_path = shell_pwd + shell_file
    start = time.time()
    subprocess.run([shell_path], shell=True)
    end = time.time()
    total = '{:.3f}'.format(end - start)
    print('Time to train: {} seconds'.format(total))
    os.chdir(PWD)

    print('Loading glove model ...')
    glove_model = load_glove_model()
    return glove_model


def vector_visualize(model, word2vec=False, glove=False, top_count=25, show=False, save=True):
    embeddings = None
    words = None
    if word2vec:
        embeddings = [model.wv[word] for word in model.wv.index_to_key][:top_count]
        words = list(model.wv.index_to_key)[:top_count]
    if glove:
        embeddings = [model[word] for word in model.index_to_key][:top_count]
        words = list(model.index_to_key)[:top_count]

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(8, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)

    for i, word in enumerate(words):
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=15)

    model_name = ''
    if word2vec:
        model_name = 'word2vec'
    if glove:
        model_name = 'glove'
    title = '{} PCA Visualization of Top {} Word Embeddings'.format(model_name, top_count)

    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    if show:
        plt.show()
    if save:
        plt.savefig(PWD + '/CSC 693 Assignment 2 Writeup/{} Top {} Vectors.png'.format(model_name, top_count))
    plt.clf()
    plt.close()


def create_labels(data_arrays):
    print('Creating data labels ...')
    labels = []
    for i in range(len(data_arrays)):
        labels += [i for _ in data_arrays[i]]
    return labels


def comment_embeddings(data, data_labels, model):
    comment_embedding_data = []
    new_labels = []
    word_embedding_data = {word: model.wv[word] for word in model.wv.index_to_key}
    for comment, label in zip(data, data_labels):
        embedding_sum = [0] * len(model.wv[model.wv.index_to_key[0]])
        word_count = 0
        for word in comment:
            if isinstance(word, str) and word in word_embedding_data:
                embedding_sum = [x + y for x, y in zip(embedding_sum, word_embedding_data[word])]
                word_count += 1
        if word_count > 0:
            embedding_avg = [x / word_count for x in embedding_sum]
            comment_embedding_data.append(embedding_avg)
            new_labels.append(label)
    return np.array(comment_embedding_data), np.array(new_labels)


def tf_pos_neg(testing_predictions, testing_labels):
    prediction_labels = [1 if prediction > 0.5 else 0 for prediction in testing_predictions]
    true_positives_list = [(p == 1 and t == 1) for p, t in zip(prediction_labels, testing_labels)]
    false_positive_list = [(p == 1 and t == 0) for p, t in zip(prediction_labels, testing_labels)]
    false_negative_list = [(p == 0 and t == 1) for p, t in zip(prediction_labels, testing_labels)]
    true_negative_list = [(p == 0 and t == 0) for p, t in zip(prediction_labels, testing_labels)]
    true_positive_sum = sum(true_positives_list)
    false_positive_sum = sum(false_positive_list)
    false_negative_sum = sum(false_negative_list)
    true_negative_sum = sum(true_negative_list)
    return true_positive_sum, false_positive_sum, false_negative_sum, true_negative_sum


def nn_eval(true_positive, false_positive, false_negative, true_negative, verbose_boolean=True):
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * (precision * recall) / (precision + recall)
    if verbose_boolean:
        print('Accuracy  = ', accuracy)
        print('Precision = ', precision)
        print('Recall    = ', recall)
        print('F1        = ', f1)
    return accuracy, precision, recall, f1


# def word2vec_nn(data, data_labels, word2vec_model, verbose_boolean=False):
#     if verbose_boolean:
#         print('Running word2vec Neural Network ...')
#         print('     Aggregating word embeddings ...')
#     comment_embedding_data, data_labels = comment_embeddings(data, data_labels, word2vec_model)
#     if verbose_boolean:
#         print('     Creating training, validation, and testing data and labels ...')
#     training_data, testing_data, training_labels, testing_labels = (
#         sklms.train_test_split(
#             comment_embedding_data,
#             data_labels,
#             test_size=0.2,
#             stratify=data_labels,
#             random_state=666
#         )
#     )
#     validation_data, testing_data = sklms.train_test_split(
#         testing_data,
#         test_size=0.5,
#         random_state=666
#     )
#     validation_labels, testing_labels = sklms.train_test_split(
#         testing_labels,
#         test_size=0.5,
#         random_state=666
#     )
#     if verbose_boolean:
#         print('     Building neural network ...')
#     neural_network = tf.keras.Sequential([
#         tf.keras.layers.Dense(
#             units=10,
#             activation='relu'
#         ),
#         tf.keras.layers.Dense(
#             units=1,
#             activation='sigmoid'
#         )
#     ])
#     print('     Compiling neural network ...')
#     neural_network.compile(
#         optimizer='adam',
#         loss='binary_crossentropy',
#         metrics=['accuracy']
#     )
#     if verbose_boolean:
#         print('     Running neural network ...')
#     neural_network.fit(
#         training_data,
#         training_labels,
#         validation_data=(
#             validation_data,
#             validation_labels
#         ),
#         epochs=10,
#         batch_size=32,
#         verbose=0
#     )
#     if verbose_boolean:
#         print('     Evaluating neural network ...')
#     testing_predictions = neural_network.predict(testing_data)
#     tp, fp, fn, tn = tf_pos_neg(testing_predictions, testing_labels)
#     accuracy, precision, recall, f1 = nn_eval(tp, fp, fn, tn, verbose_boolean=verbose_boolean)
#     return accuracy, precision, recall, f1
#
#
# def w2v_nn_testing(data, data_labels, max_vector_size):
#     print('Running Neural Network for vector sizes from 1 to {}'.format(max_vector_size))
#     size_list = [i + 1 for i in range(max_vector_size)]
#     accuracy_list = []
#     precision_list = []
#     recall_list = []
#     f1_list = []
#     for vector_size in size_list:
#         print('Current vector size =', vector_size)
#         word2vec_testing = word2vec_cbow(data, vector_size=vector_size, window=5, min_count=1, verbose=False)
#         accuracy, precision, recall, f1 = word2vec_nn(data, data_labels, word2vec_testing, verbose_boolean=False)
#         accuracy_list.append(100*accuracy)
#         precision_list.append(100*precision)
#         recall_list.append(100*recall)
#         f1_list.append(100*f1)
#     fig, ax = plt.subplots()
#     ax.plot(size_list, accuracy_list, marker='o', label='Accuracy', zorder=4)
#     ax.plot(size_list, precision_list, marker='o', label='Precision', zorder=3)
#     ax.plot(size_list, recall_list, marker='o', label='Recall', zorder=2)
#     ax.plot(size_list, f1_list, marker='o', label='F1 Score', zorder=1)
#     plt.xlabel('Vector Size')
#     plt.ylabel('Percentage')
#     plt.title('Word2Vec Metrics vs. Vector Size')
#     plt.legend()
#     plt.show()
#     fig.savefig(PWD + '/CSC 693 Assignment 2 Writeup/w2v_nn_vector_size_plot.png')
#     plt.clf()
#     plt.close()
