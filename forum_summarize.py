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


def create_comment_list(post):
    sorted_ids = sorted(post.comments, key=lambda comment: comment.score, reverse=True)
    reddit_thread = [[post.title, post.selftext]]
    comment_list = []
    main_count = -1  # Start main comment count at 0
    for comment_id in sorted_ids:
        if comment_id.body != '[removed]':
            main_count += 1
            sub_count = 0
            comment_list.append([main_count, sub_count, comment_id.score, comment_id.id, ''])
            if comment_id.replies:
                sub_comments = create_subcomment_list(comment_id.replies, main_count)
                comment_list.extend(sub_comments)
    reddit_thread.extend(comment_list)
    return reddit_thread


def create_subcomment_list(replies):
    sub_comments = []
    for comment in replies:
        if comment.body != '[removed]':
            sub_comments.append([comment.score, comment.id, ''])
            if comment.replies:
                sub_sub_comments = create_subcomment_list(comment.replies, main_count)
                sub_comments.extend(sub_sub_comments)
    return sub_comments


def get_replies(selection):
    for selected_comment in selection.replies:
        if selected_comment.body != '[removed]' and selected_comment.author != 'AutoModerator':
            print(str(selected_comment.score) + ', ' + selected_comment.body[0:20] + ' ... ')
            if selected_comment.replies:
                get_replies(selected_comment)


def create_dataset(post_url):
    print('Loading data from URL')
    post = REDDIT.submission(url=post_url)
    post.comment_sort = "top"
    post.comments.replace_more(limit=None)
    # print('Loading {} commments past \'MoreComments\''.format(post.num_comments))
    pprint.pprint(vars(post))
    print(len(post.comments.list()))
    comment_list = []
    comment_count = 0
    for selected_comment in post.comments:
        if selected_comment.body != '[removed]' and selected_comment.author != 'AutoModerator':
            print(str(selected_comment.score) + ', ' + selected_comment.body[0:20] + ' ... ')
            if selected_comment.replies:
                get_replies(selected_comment)
                input('wait')
            print('-----------------------------------')
            comment_list.extend(selected_comment.replies)
    input()
    reddit_thread = create_comment_list(post)
    for item in reddit_thread[1:30]:
        print(item)


create_dataset(askhistorians[1])
# create_dataset(science[3])
