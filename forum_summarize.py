from dataset_links import askreddit
from dataset_links import science
import praw_functions as pfx


reddit_post, reddit_thread = pfx.create_dataset(science[2])
for a in reddit_thread:
    pfx.parse_comment_structure(a, verbose=True)
    print('\n-----')
# pprint.pprint(vars(reddit_post))
