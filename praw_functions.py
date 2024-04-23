import os
import time
import pickle
import statistics
import numpy as np
import praw
# import torch
from operator import itemgetter
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# from transformers import BartTokenizer, BartForConditionalGeneration
# from transformers import pipeline
import text_functions as tfx

THREAD_PARAGRAPH = []  # list of all unstemmed comments in thread converted to a string, each comment separated by '. '
MAIN_THREADS_PARAGRAPHS = []  # list of lists of all unstemmed thread paragraphs converted to a string
MAIN_THREAD_COMMENTS_STEMMED = []  # list of all comments in thread converted to a string, each comment separated by '. '
MAIN_THREADS_COMMENTS_STEMMED = []  # list of lists of all thread paragraphs converted to a string
global COMMENT_DEPTH_FILTER
MAIN_THREAD_DEPTHS = []
global WORD_FREQ_SUM
global TOP_POST_WORDS_100
print('Loading BART Tokenizer ...')
TOKENIZER = AutoTokenizer.from_pretrained("Mr-Vicky-01/Bart-Finetuned-conversational-summarization")
# TOKENIZER = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
print('Loading BART Model ...')
MODEL = AutoModelForSeq2SeqLM.from_pretrained("Mr-Vicky-01/Bart-Finetuned-conversational-summarization")


# https://huggingface.co/Mr-Vicky-01/Bart-Finetuned-conversational-summarization
# MODEL = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")
# https://huggingface.co/sshleifer/distilbart-cnn-12-6


def initialize_praw():
    base_path = os.path.dirname(__file__)
    file_path = os.path.abspath(os.path.join(base_path, '..', 'reddit_praw_keys.txt'))
    with open(file_path, 'r') as key_file:
        for key in key_file:
            key_list = key.split(', ')
    reddit_praw = praw.Reddit(
        client_id=key_list[0],
        client_secret=key_list[1],
        password=key_list[2],
        user_agent=key_list[3],
        username=key_list[4],
    )
    return reddit_praw


REDDIT = initialize_praw()
AUTOMOD = 'AutoModerator'


def load_reddit_post(post_url):
    print('Loading data from URL ...')
    post = REDDIT.submission(url=post_url)
    print('Sorting data by \'top\' ...')
    post.comment_sort = "top"
    start = time.time()
    print('Loading full data ...')
    post.comments.replace_more(limit=None)
    end = time.time()
    runtime = end - start
    print('Time to load full data = {} minutes, {} seconds'.format(int(runtime // 60), round(runtime % 60, 3)))
    return post


def get_comments(post):
    comment_list = []
    for comment in post.comments:
        if (
                comment.body != '[removed]' and
                comment.body != '[deleted]' and
                comment.author != 'AutoModerator'
        ):
            tfx.preprocess(comment)
            comment_list.append(
                [
                    comment.id
                ]
            )
            get_replies(comment, comment_list)
    return comment_list


def get_replies(comment, comment_list):
    if comment.replies:
        for comment in comment.replies:
            if (
                    comment.author != 'AutoModerator'
            ):
                if (
                        comment.body != '[removed]' or
                        comment.body == '[deleted]'
                ):
                    tfx.preprocess(comment)
                elif (
                        comment.body == '[removed]' or
                        comment.body == '[deleted]'
                ):
                    tfx.COMMENT_LIST_DICT.update({comment.id: ['']})
                comment_list[-1].append(
                    [
                        comment.id
                    ]
                )
                get_replies(comment, comment_list[-1])


def create_dataset(post_url):
    global WORD_FREQ_SUM
    global TOP_POST_WORDS_100

    print('Selected post URL: {}'.format(post_url))
    post_id = post_url.split('comments/')[-1].split('/')[0]
    pkl_exists = False
    post = None

    print('Searching for existing \'.pkl\' file of post data ... ')
    for list_item in os.listdir(os.getcwd()):
        if list_item == post_id + '.pkl':
            pkl_exists = True
    if pkl_exists:
        print('\'.pkl\' file found! :D')
        with open(post_id + '.pkl', 'rb') as file:
            post, tfx.POST_DICT, tfx.WORD_FREQ, tfx.COMMENT_LIST_DICT, tfx.COMMENT_DEPTH_DICT = pickle.load(file)
    elif not pkl_exists:
        print('\'.pkl\' file not found :(')
        post = load_reddit_post(post_url)
        print('Saving data to \'.pkl\' file ...')
        with open(post_id + '.pkl', 'wb') as file:
            pickle.dump((post, tfx.POST_DICT, tfx.WORD_FREQ, tfx.COMMENT_LIST_DICT, tfx.COMMENT_DEPTH_DICT), file)

    print('Forming comment list ...')
    start = time.time()
    comment_list = get_comments(post)
    end = time.time()
    runtime = end - start
    print('Time to form comment list = {} minutes, {} seconds'.format(int(runtime // 60), round(runtime % 60, 3)))

    tfx.WORD_FREQ = {word: freq for word, freq in reversed(sorted(tfx.WORD_FREQ.items(), key=lambda item: item[1]))}
    WORD_FREQ_SUM = sum(sorted(tfx.WORD_FREQ.values(), reverse=True)[:100])
    TOP_POST_WORDS_100 = dict(sorted(tfx.WORD_FREQ.items(), key=itemgetter(1), reverse=True)[:100])
    return post, comment_list


def transform_comment(comment_id, translate_stemmed=False, to_paragraph=False):
    comment_sentences_list = tfx.COMMENT_LIST_DICT[comment_id]
    transformed_comment = []
    for sentence_word_list in comment_sentences_list:
        temp = []
        for word in sentence_word_list:
            if not translate_stemmed:
                temp.append(word)
            if translate_stemmed:
                temp.append(tfx.POST_DICT[word])
        transformed_comment.append(temp)
    output = transformed_comment
    if to_paragraph:
        comment_paragraph = tfx.string_list_to_paragraph(transformed_comment, comment_paragraph='')
        output = comment_paragraph
    return output


def parse_operations(comment_id):
    comment_text = transform_comment(comment_id, translate_stemmed=False, to_paragraph=False)
    MAIN_THREAD_COMMENTS_STEMMED.extend(comment_text)
    comment_text = transform_comment(comment_id, translate_stemmed=True, to_paragraph=True)
    THREAD_PARAGRAPH.append(comment_text)


def parse_comment_structure(thread, marker='|', comment_level=1, verbose=None):
    if isinstance(thread, list):
        for index, item in enumerate(thread):
            if not isinstance(item, list):
                if isinstance(item, str):
                    tfx.COMMENT_DEPTH_DICT.update({item: comment_level})
                    if (
                            tfx.COMMENT_LIST_DICT[item] != [''] and
                            comment_level <= COMMENT_DEPTH_FILTER
                    ):
                        MAIN_THREAD_DEPTHS.append(comment_level)
                        parse_operations(item)
                if verbose:
                    # print items in list (upvotes, then comment id string)
                    print(str(item) + ', ', end='')
            elif isinstance(item, list):
                if verbose:
                    # print new line with markers and comment level
                    print('')
                    print(marker + ' {}: '.format(comment_level), end='')
                parse_comment_structure(item, marker=marker + '|', comment_level=len(marker) + 1, verbose=verbose)


def comment_structure_manipulation(reddit_post, reddit_thread):
    global TOP_POST_WORDS_100
    global MAIN_THREAD_DEPTHS
    main_thread_paragraph_list = []

    median_upvotes = upvote_statistics(reddit_post, reddit_thread)
    print('Median number of upvotes per main comment = {}'.format(median_upvotes))
    print('Parsing comment structure ...')
    for index, main_comment in enumerate(reddit_thread):
        THREAD_PARAGRAPH.clear()
        MAIN_THREAD_COMMENTS_STEMMED.clear()
        MAIN_THREAD_DEPTHS.clear()
        tfx.COMMENT_DEPTH_DICT.update({main_comment[0]: 0})
        parse_comment_structure(main_comment, verbose=False)
        percent_top_words = top_word_percentage()
        if (
                len(MAIN_THREAD_DEPTHS) > 3 and
                percent_top_words > 0.10 and
                tfx.COMMENT_UPVOTE_DICT[main_comment[0]] >= median_upvotes
        ):
            print(
                'Main Thread #{} has {} upvotes with {} comments\n     and contains {}% of top 100 words\n'.format(
                    index,
                    len(MAIN_THREAD_DEPTHS),
                    tfx.COMMENT_UPVOTE_DICT[main_comment[0]],
                    int(percent_top_words * 100)
                )
            )
            create_plot_embeddings(index, reddit_post)
            MAIN_THREADS_COMMENTS_STEMMED.extend(MAIN_THREAD_COMMENTS_STEMMED)

            # Combine all data into one string for entire block of text summary
            main_thread_string = ''
            main_thread_paragraph_list.append(THREAD_PARAGRAPH)
            for thread_paragraph_string in main_thread_paragraph_list:
                main_thread_string = tfx.string_list_to_string(thread_paragraph_string, main_thread_string)
            MAIN_THREADS_PARAGRAPHS.append(main_thread_string)
    print('Done Parsing!')


def top_word_percentage():
    thread_top_words = []
    unique_stemmed_words = []
    for comment in MAIN_THREAD_COMMENTS_STEMMED:
        unique_stemmed_words.extend(comment)
    unique_stemmed_words = np.ndarray.tolist(np.unique(unique_stemmed_words))
    for top_thread_word in unique_stemmed_words:
        if TOP_POST_WORDS_100.__contains__(top_thread_word):
            thread_top_words.append(top_thread_word)
    return len(thread_top_words) / len(TOP_POST_WORDS_100)


def upvote_statistics(reddit_post, reddit_thread):
    print('Filtering main comments by upvotes ...')
    upvote_list = []
    for index, main_comment in enumerate(reddit_thread):
        upvotes = tfx.COMMENT_UPVOTE_DICT[main_comment[0]]
        if upvotes > 5:
            upvote_list.append(upvotes)
    # mean = sum(upvote_list)/len(upvote_list)
    # variance = sum([((x - mean) ** 2) for x in upvote_list]) / len(upvote_list)
    # standard_deviation = np.sqrt(variance)
    quartile_1 = np.percentile(upvote_list, 25)
    quartile_3 = np.percentile(upvote_list, 75)
    interquartile_range = quartile_3 - quartile_1
    lower_bound = quartile_1 - 1.5 * interquartile_range
    upper_bound = quartile_3 + 1.5 * interquartile_range
    median_upvotes = statistics.median(upvote_list)

    outliers = [x for x in upvote_list if x < lower_bound or x > upper_bound]

    # Remove outliers
    upvote_list_no_outliers = [x for x in upvote_list if lower_bound <= x <= upper_bound]
    plt.boxplot(upvote_list_no_outliers, vert=False)
    post_title = reddit_post.title
    post_title = '\"{}\"'.format(post_title)
    if len(reddit_post.title) > 25:
        post_title = '\"{} ...\"'.format(post_title[:25])
    plt.legend(["Outliers: " + ", ".join(map(str, outliers))], loc="upper right")
    plt.title('Boxplot of Upvotes for Reddit Post:\n{}'.format(post_title))
    plt.xlabel('Upvotes (upvotes greater than 5)')
    plt.yticks([])
    save_name = '/code_output/Post {}, upvotes boxplot.png'.format(reddit_post.id)
    plt.savefig(os.getcwd() + save_name)
    return median_upvotes


def create_plot_embeddings(index, reddit_post):
    # print('Running word2Vec to generate word embeddings ...')
    vector_count = 25
    w2v_model = Word2Vec(MAIN_THREAD_COMMENTS_STEMMED, vector_size=100, window=5, min_count=1, sg=0, epochs=30)
    embeddings = [w2v_model.wv[word] for word in w2v_model.wv.index_to_key][:vector_count]
    words = list(w2v_model.wv.index_to_key)[:vector_count]
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)

    # print('Saving plots of word embeddings ...')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for word, vector in zip(words, embeddings_3d):
        ax.scatter(vector[0], vector[1], vector[2], color='b')
        ax.text(vector[0], vector[1], vector[2], tfx.POST_DICT[word], color='black')
        ax.plot([0, vector[0]], [0, vector[1]], [0, vector[2]], color='gray', alpha=0.3)
    post_title = reddit_post.title
    post_title = '\"{}\"'.format(post_title)
    if len(reddit_post.title) > 25:
        post_title = '\"{} ...\"'.format(post_title[:25])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    legend_elements = ['{}: {}'.format(i + 1, tfx.POST_DICT[word]) for i, word in enumerate(words)]
    legend = ax.legend(
        labels=legend_elements,
        title='Top Words in\n'
              'Descending Order',
        loc='center left',
        bbox_to_anchor=(1.0, 0.5)
    )
    plt.title(
        '3-Component PCA of Word Embeddings\n'
        'for Top {} Words in Main Comment Thread #{}\n on Post {}'.format(vector_count, index + 1, post_title)
    )
    plt.savefig(
        os.getcwd() + '/code_output/Post {}, 3D Embedding Plot Main Thread {}.png'.format(reddit_post.id, index + 1),
        bbox_extra_artists=(legend,),
        bbox_inches='tight'
    )


def generate_summary(input_string):
    inputs = TOKENIZER([input_string], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = MODEL.generate(inputs['input_ids'], max_new_tokens=100, do_sample=False)
    summary = TOKENIZER.decode(summary_ids[0], skip_special_tokens=True)
    return summary
