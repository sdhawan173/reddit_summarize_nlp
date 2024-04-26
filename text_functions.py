import os
import string
import nltk
import re
from cleantext import clean
from nltk.corpus import stopwords

POST_DICT = {}  # {lemmatized word: original word (1st variation only)}
FILT_DICT = {}  # same as post dict, but filtered words only
POST_FREQ = {}  # {lemmatized word: lemm. word frequency}
FILT_FREQ = {}
COMMENT_LIST_DICT = {}  # {comment.id: list preprocessed comment sentences}
COMMENT_UPVOTE_DICT = {}  # {comment.id: comment.score}
COMMENT_DEPTH_DICT = {}  # {comment.id: depth of comment}

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
PUNCTUATION.append('\'\'')
PUNCTUATION.append('\"')
PUNCTUATION.append('\n')
PUNCTUATION.append('..')
PUNCTUATION.append('...')
PUNCTUATION.append('....')
PUNCTUATION.append('.....')
PUNCTUATION.append('......')
PWD = os.getcwd()

STEMMER = nltk.PorterStemmer()


def string_list_to_paragraph(transformed_comment, comment_paragraph):
    for sentence_word_list in transformed_comment:
        for word in sentence_word_list:
            comment_paragraph += ' ' + word
        comment_paragraph += '.'
    return comment_paragraph


def string_list_to_string(string_list, list_as_string, end_append=False):
    temp = None
    if end_append:
        temp = list_as_string
        list_as_string = ''
    for string_item in string_list:
        list_as_string += string_item
    if end_append:
        list_as_string += temp
    return list_as_string


def remove_urls(text):
    # Define the regex pattern to match URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    # Replace all matches of the URL pattern with an empty string
    text_without_urls = url_pattern.sub('', text)
    return text_without_urls


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def process_string(input_string):
    processed = []
    for word in tokenize(input_string):
        if (
                not PUNCTUATION.__contains__(word.lower()) and
                not STOP_WORDS.__contains__(word.lower())
        ):
            word_lower = word.lower()
            stemmed_word = STEMMER.stem(word_lower)
            processed.append(stemmed_word)
    return processed


def split_words(sentence):
    """
    Splits each comment from list of comments into words
    :param sentence: sentence string (punctuation not necessary)
    :return: list of lists of comments split into words
    """
    preprocessed_words = []
    if sentence.__contains__('http'):
        sentence = remove_urls(sentence)
    text_split_word = tokenize(sentence)
    for word in text_split_word:
        if (
                not PUNCTUATION.__contains__(word.lower()) and
                not STOP_WORDS.__contains__(word.lower()) and
                clean(word, no_emoji=True) != ''
        ):
            word_lower = word.lower()
            stemmed_word = STEMMER.stem(word_lower)
            preprocessed_words.append(stemmed_word)
            if stemmed_word not in POST_DICT.keys():
                POST_DICT.update({stemmed_word: word_lower})
                POST_FREQ.update({stemmed_word: 1})
            elif stemmed_word in POST_DICT.keys():
                POST_FREQ[stemmed_word] = POST_FREQ.get(stemmed_word) + 1
    return preprocessed_words


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
    COMMENT_UPVOTE_DICT.update({comment.id: comment.score})
    COMMENT_LIST_DICT.update({comment.id: split_comment})


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
