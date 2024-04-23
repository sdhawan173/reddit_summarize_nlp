from dataset_links import askreddit
from dataset_links import science
import praw_functions as pfx
import text_functions as tfx


# two experiements
# 1 - all data sequentially into model, let model summarize conversation
#     - sort main comments by top, output short sentence or sentences
# 2 - how to organize conversation

reddit_post, reddit_thread = pfx.create_dataset(science[2])
pfx.COMMENT_DEPTH_FILTER = 5
pfx.comment_structure_manipulation(reddit_post, reddit_thread)
entire_post_string = tfx.string_list_to_string(pfx.MAIN_THREADS_PARAGRAPHS, '')
entire_post_string += reddit_post.title
entire_summary = pfx.generate_summary(entire_post_string)
print('Entire Summary\n', entire_summary)

paragraph_summaries = []
for index, paragraph in enumerate(pfx.MAIN_THREADS_PARAGRAPHS):
    summary = pfx.generate_summary(paragraph)
    paragraph_summaries.append(summary)
    print('\nSummary {}\n{}'.format(index, paragraph_summaries[-1]))

# print('Running Facebook BART Large CNN ...')
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# # https://huggingface.co/facebook/bart-large-cnn
# summary = summarizer(entire_post_string, max_length=130, min_length=30, do_sample=False)
