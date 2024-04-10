import os
import time
from tqdm import tqdm
import pickle
import praw
import pprint
from dataset_links import askreddit
from dataset_links import science
import text_file_operations as tfo
import text_functions as tfx


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


def get_moderators(subreddit_name):
    return REDDIT.subreddit(subreddit_name).moderator()


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


def preprocess(comment):
    comment_body = comment.body
    split_comment = []
    comment_sentence_split = tfx.split_sentence(comment_body)
    for sentence in comment_sentence_split:
        sentence_split_words = tfx.split_words(sentence)
        split_comment.append(sentence_split_words)
    return split_comment


def get_comments(post):
    comment_list = []
    for comment in post.comments:
        if (
                comment.body != '[removed]' and
                comment.author != 'AutoModerator' and
                comment.score >= 5
        ):
            comment_body = preprocess(comment)
            comment_list.append(
                [
                    comment.score,
                    comment.id,
                    comment_body,
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
                comment_body = ''
                if comment.body != '[removed]':
                    comment_body = preprocess(comment)
                comment_list[-1].append(
                    [
                        comment.score,
                        comment.id,
                        comment_body
                    ]
                )
                get_replies(comment, comment_list[-1])


def create_dataset(post_url):
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
            post = pickle.load(file)
    elif not pkl_exists:
        print('\'.pkl\' file not found :(')
        post = load_reddit_post(post_url)
        print('Saving data to \'.pkl\' file ...')
        with open(post_id + '.pkl', 'wb') as file:
            pickle.dump(post, file)

    print('Forming comment list ...')

    start = time.time()
    comment_list = get_comments(post)
    end = time.time()
    runtime = end - start
    print('Time to form comment list = {} minutes, {} seconds'.format(int(runtime // 60), round(runtime % 60, 3)))

    tfx.WORD_FREQ = {word: freq for word, freq in reversed(sorted(tfx.WORD_FREQ.items(), key=lambda item: item[1]))}
    return post, comment_list


def parse_comment_structure(thread, marker='|', comment_level=1, verbose=None):
    if isinstance(thread, list):
        for index, item in enumerate(thread):
            if not isinstance(item, list):
                if verbose:
                    print(str(item) + ', ', end='')
            elif isinstance(item, list):
                if verbose:
                    print('')
                    print(marker + ' {}: '.format(comment_level), end='')
                parse_comment_structure(item, marker=marker + '|', comment_level=len(marker) + 1, verbose=verbose)


reddit_post, reddit_thread = create_dataset(science[2])
for a in reddit_thread:
    parse_comment_structure(a, verbose=True)
    print('\n-----')
# pprint.pprint(vars(reddit_post))
