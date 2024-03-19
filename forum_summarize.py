import os
import praw


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


print(REDDIT.user.me())
