import os
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


def load_reddit_post(url_string):
    return REDDIT.submission(url=url_string)


def get_replies(selection, sub_comment_list, verbose=None):
    for selected_comment in selection.replies:
        if selected_comment.body != '[removed]' and selected_comment.author != 'AutoModerator':
            sub_comment_list.append([selected_comment.score, selected_comment.id, 'selected_comment.body'])
            if verbose:
                print(str(sub_comment_list[-1][0]) + ', ' + sub_comment_list[-1][1][0:20] + ' ... ')
            if selected_comment.replies:
                sub_sub_comment_list = []
                get_replies(selected_comment, sub_sub_comment_list, verbose=verbose)
                sub_comment_list.append(sub_sub_comment_list)
    return sub_comment_list


def create_dataset(post_url, verbose=None):
    print('Loading data from URL')
    post = REDDIT.submission(url=post_url)
    post.comment_sort = "top"
    post.comments.replace_more(limit=None)
    comment_list = []
    for selected_comment in post.comments:
        if selected_comment.body != '[removed]' and selected_comment.author != 'AutoModerator':
            comment_list.append([selected_comment.score, selected_comment.id, 'selected_comment.body'])
            if verbose:
                print('-----New Main Comment')
                print(str(comment_list[-1][0]) + ', ' + comment_list[-1][1][0:20] + ' ... ')
            if selected_comment.replies:
                sub_comment_list = []
                get_replies(selected_comment, sub_comment_list, verbose=verbose)
                comment_list[-1].append(sub_comment_list)
    return post, comment_list


def check_comment_structure(thread, space=' '):
    print('')
    if isinstance(thread, list):
        print(space, end='')
        for item in thread:
            if not isinstance(item, list):
                print(str(item) + ', ', end='')
            elif isinstance(item, list):
                check_comment_structure(item, space=space + ' ')


reddit_post, comment_thread = create_dataset(askhistorians[1], verbose=False)
check_comment_structure(comment_thread)
# pprint.pprint(vars(reddit_post))
# create_dataset(science[3])
