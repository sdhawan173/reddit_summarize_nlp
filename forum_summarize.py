from dataset_links import askreddit
from dataset_links import science
import praw_functions as pfx
import text_functions as tfx


reddit_post, reddit_thread = pfx.create_dataset(science[2])
# pprint.pprint(vars(reddit_post))
for main_comment in reddit_thread:
    tfx.COMMENT_DEPTH_DICT.update({main_comment[1]: 0})
    pfx.parse_comment_structure(main_comment, verbose=True)
    print('\n-----')
print(tfx.POST_DICT)
print(tfx.WORD_FREQ)
print(tfx.COMMENT_DEPTH_DICT)
print(tfx.COMMENT_LIST_DICT)
