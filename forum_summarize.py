import os
import time
from tqdm import tqdm
import pickle
import praw
import pprint
from dataset_links import askreddit
from dataset_links import askhistorians
from dataset_links import science


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


def get_replies(comment_list, selection=None, iter_list=None):
    if iter_list is None:
        iter_list = selection.replies
    for selected_comment in iter_list:
        if selected_comment.body != '[removed]' and selected_comment.author != 'AutoModerator':
            comment_list.append([selected_comment.score, selected_comment.id, 'selected_comment.body'])
            if selected_comment.replies:
                sub_sub_comment_list = []
                get_replies(sub_sub_comment_list, selection=selected_comment)
                comment_list.append(sub_sub_comment_list)
    return comment_list


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
    comment_list = []
    print('Forming comment list ...')
    start = time.time()
    comment_list = get_replies(comment_list, iter_list=post.comments)
    end = time.time()
    runtime = end - start
    print('Time to form comment list = {} minutes, {} seconds'.format(int(runtime // 60), round(runtime % 60, 3)))
    return post, comment_list


def parse_comment_structure(thread, space='|', verbose=None):
    if isinstance(thread, list):
        print_count = 0
        for index, item in enumerate(thread):
            if not isinstance(item, list):
                if print_count == 2:
                    if verbose:
                        print(str(item))
                else:
                    print_count += 1
                    if verbose:
                        print(str(item) + ', ', end='')
            elif isinstance(item, list):
                if verbose:
                    print(space, end='')
                parse_comment_structure(item, space=space + '|', verbose=verbose)


reddit_post, comment_thread = create_dataset(science[1])
parse_comment_structure(comment_thread, verbose=True)
# pprint.pprint(vars(reddit_post))
# create_dataset(science[3])
