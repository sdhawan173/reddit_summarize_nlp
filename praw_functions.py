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
FILT_ALL_THREADS_PARAGRAPHS = []  # list of lists of all unstemmed thread paragraphs converted to a string
MAIN_THREAD_COMMENTS_STEMMED = []  # list of all comments in thread converted to string, each comment separated by '. '
FILT_ALL_COMMENTS_STEMMED = []  # list of lists of all thread paragraphs converted to a string
global COMMENT_DEPTH_FILTER
MAIN_THREAD_DEPTHS = []
global TOP_POST_WORDS_100
global PERCENT_TOP_WORDS
global UPVOTE_CRITERION
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
    global TOP_POST_WORDS_100

    print('Selected post URL: {}'.format(post_url))
    post_id = post_url.split('comments/')[-1].split('/')[0]
    pkl_exists = False
    post = None

    print('Searching for existing \'.pkl\' file of post data ... ')
    pkl_directory = os.getcwd() + '/pkl_files/'
    for list_item in os.listdir(pkl_directory):
        if list_item == post_id + '.pkl':
            pkl_exists = True
    if pkl_exists:
        print('\'.pkl\' file found! :D')
        with open(pkl_directory + post_id + '.pkl', 'rb') as file:
            post = pickle.load(file)
    elif not pkl_exists:
        print('\'.pkl\' file not found :(')
        post = load_reddit_post(post_url)
        print('Saving data to \'.pkl\' file ...')
        with open(pkl_directory + post_id + '.pkl', 'wb') as file:
            pickle.dump(post, file)
    print('Forming comment list ...')
    start = time.time()
    comment_list = get_comments(post)
    end = time.time()
    runtime = end - start
    print('Time to form comment list = {} minutes, {} seconds'.format(int(runtime // 60), round(runtime % 60, 3)))

    tfx.POST_FREQ = {word: freq for word, freq in reversed(sorted(tfx.POST_FREQ.items(), key=lambda item: item[1]))}
    TOP_POST_WORDS_100 = dict(sorted(tfx.POST_FREQ.items(), key=itemgetter(1), reverse=True)[:100])
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
                    MAIN_THREAD_DEPTHS.append(comment_level)
                    if (
                            tfx.COMMENT_LIST_DICT[item] != [''] and
                            comment_level <= COMMENT_DEPTH_FILTER
                    ):
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
    global UPVOTE_CRITERION
    global PERCENT_TOP_WORDS

    all_thread_comment_sizes = []
    main_thread_paragraph_list = []
    filtered_thread_nums = []
    w2v_models = []
    median_upvotes = upvote_stats_and_boxplot(reddit_post, reddit_thread)
    UPVOTE_CRITERION = median_upvotes
    print('The upvote cutoff is the median number\n'
          'of upvotes of all main comments: {}\n'.format(median_upvotes))
    print('Parsing comment structure ...')
    print('     Analyzing filtered dataset ...')
    for index, main_comment in enumerate(reddit_thread):
        # Clear temp global lists
        THREAD_PARAGRAPH.clear()
        MAIN_THREAD_COMMENTS_STEMMED.clear()
        MAIN_THREAD_DEPTHS.clear()

        # add initial main thread comment depth level to depth dictionary for comment ID
        tfx.COMMENT_DEPTH_DICT.update({main_comment[0]: 0})

        # Recursive Parsing
        parse_comment_structure(main_comment, verbose=False)

        # Calculate percentage of top words in entire post for words in current main thread
        PERCENT_TOP_WORDS = top_word_percentage(TOP_POST_WORDS_100)
        if (
                len(MAIN_THREAD_DEPTHS) > 5 and
                PERCENT_TOP_WORDS > 0.10 and
                tfx.COMMENT_UPVOTE_DICT[main_comment[0]] >= UPVOTE_CRITERION
        ):
            print(
                '     Main Thread #{}, id {}:\n'
                '          {} upvotes on the initial comment\n'
                '          {} sub-comments at different comment level depths\n'
                '          max comment depth of {} (filtered up to max depth of {})\n'
                '          contains {}% of top 100 words in the reddit post'
                '\n'.format(index + 1, main_comment[0],
                            tfx.COMMENT_UPVOTE_DICT[main_comment[0]],
                            len(MAIN_THREAD_DEPTHS),
                            max(MAIN_THREAD_DEPTHS), COMMENT_DEPTH_FILTER,
                            int(PERCENT_TOP_WORDS * 100)
                            )
            )

            # add number ordering of current filtered thread to list
            filtered_thread_nums.append(index + 1)

            # run word2vec to generate embeddings for current thread and add to list
            w2v_models.append(
                Word2Vec(
                    MAIN_THREAD_COMMENTS_STEMMED,
                    vector_size=100,
                    window=5,
                    min_count=1,
                    sg=0,
                    epochs=30
                )
            )

            # concatenate temp global list of all stemmed comment lists to global list of all stemmed comments
            FILT_ALL_COMMENTS_STEMMED.extend(MAIN_THREAD_COMMENTS_STEMMED)

            # add stemmed words to new dictionary containing stemmed words of filtered threads
            for stemmed_comment in MAIN_THREAD_COMMENTS_STEMMED:
                for stemmed_word in stemmed_comment:
                    if stemmed_word not in tfx.FILT_DICT.keys():
                        tfx.FILT_DICT.update({stemmed_word: tfx.POST_DICT[stemmed_word]})
                        tfx.FILT_FREQ.update({stemmed_word: 1})
                    elif stemmed_word in tfx.FILT_DICT.keys():
                        tfx.FILT_FREQ[stemmed_word] = tfx.FILT_FREQ.get(stemmed_word) + 1

            # Combine all data into one string for entire block of text summary
            main_thread_string = ''
            main_thread_paragraph_list.append(THREAD_PARAGRAPH)
            for thread_paragraph_string in main_thread_paragraph_list:
                main_thread_string = tfx.string_list_to_string(thread_paragraph_string, main_thread_string)
            FILT_ALL_THREADS_PARAGRAPHS.append(main_thread_string)
            all_thread_comment_sizes.append(len(MAIN_THREAD_DEPTHS))

    for model, index in zip(w2v_models, filtered_thread_nums):
        create_plot_embeddings(model, index, reddit_post, len(MAIN_THREAD_DEPTHS))
    data_pie_chart(filtered_thread_nums, all_thread_comment_sizes, reddit_post)
    print('Done Parsing!')


def top_word_percentage(top_words):
    thread_top_words = []
    unique_stemmed_words = []
    for comment in MAIN_THREAD_COMMENTS_STEMMED:
        unique_stemmed_words.extend(comment)
    unique_stemmed_words = np.ndarray.tolist(np.unique(unique_stemmed_words))
    for top_thread_word in unique_stemmed_words:
        if top_words.__contains__(top_thread_word):
            thread_top_words.append(top_thread_word)
    return len(thread_top_words) / len(top_words)


def data_pie_chart(filtered_thread_nums, all_thread_comment_sizes, reddit_post):
    comment_sum = sum(all_thread_comment_sizes)
    labels = [
        'Main Thread #{0}:\n{1:.2f}%, {2} comments'.format(num, 100 * size / comment_sum, size)
        for num, size in zip(filtered_thread_nums, all_thread_comment_sizes)
    ]
    pie = plt.pie(all_thread_comment_sizes)
    legend = plt.legend(pie[0], labels, bbox_to_anchor=(1.15, 0.5), loc="center right", fontsize=10,
                        bbox_transform=plt.gcf().transFigure)
    plt.title('Filtered Main Comment Threads,\n'
              'Contributions to the Dataset ({} total comments)'.format(comment_sum))
    plt.savefig(os.getcwd() + '/code_output/Post {}, Pie Chart.png'.format(reddit_post.id),
                bbox_extra_artists=(legend,),
                bbox_inches='tight')


def upvote_stats_and_boxplot(reddit_post, reddit_thread):
    print('Filtering main comments by upvotes ...')
    upvote_list = []
    for index, main_comment in enumerate(reddit_thread):
        upvotes = tfx.COMMENT_UPVOTE_DICT[main_comment[0]]
        if tfx.COMMENT_UPVOTE_DICT[main_comment[0]] >= 5:
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
    plt.close()
    return median_upvotes


def create_plot_embeddings(w2v_model, index, reddit_post, num_comments):
    # print('Running word2Vec to generate word embeddings ...')
    vector_count = 25
    embeddings = [w2v_model.wv[word] for word in w2v_model.wv.index_to_key][:vector_count]
    words = list(w2v_model.wv.index_to_key)[:vector_count]
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)

    # print('Saving plots of word embeddings ...')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for word, vector in zip(words, embeddings_3d):
        ax.scatter(vector[0], vector[1], vector[2], color='b')
        ax.text(vector[0], vector[1], vector[2], tfx.FILT_DICT[word], color='black')
        ax.plot([0, vector[0]], [0, vector[1]], [0, vector[2]], color='gray', alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    legend_elements = ['{}: {}'.format(i + 1, tfx.FILT_DICT[word]) for i, word in enumerate(words)]
    legend = ax.legend(
        labels=legend_elements,
        title='Top Words in\n'
              'Descending Order',
        loc='center left',
        bbox_to_anchor=(1.0, 0.5)
    )
    plt.title(
        '3-Component PCA of Word Embeddings:\n'
        'Top {} Words in Main Comment Thread #{}\n'
        '({} total comments in thread)'.format(
            vector_count,
            index + 1,
            num_comments,
        )
    )
    caption = plt.figtext(
        0.5,
        0.1,
        'Reddit Post ID \"{}\"'.format(
            reddit_post.id
        ),
        wrap=True,
        va='center',
        ha='center',
        fontsize=12
    )
    plt.savefig(
        os.getcwd() + '/code_output/Post {}, 3D Embedding Plot Main Thread {}.png'.format(reddit_post.id, index + 1),
        bbox_extra_artists=(legend, caption),
        bbox_inches='tight'
    )
    plt.close()


def generate_summary(input_string):
    inputs = TOKENIZER([input_string], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = MODEL.generate(inputs['input_ids'], max_new_tokens=100, do_sample=False)
    summary = TOKENIZER.decode(summary_ids[0], skip_special_tokens=True)
    return summary
