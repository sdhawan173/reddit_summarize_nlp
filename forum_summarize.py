from rouge import Rouge
from dataset_links import askreddit
from dataset_links import science
import praw_functions as pfx
import text_functions as tfx


def evaluate(summary_string, title, body=None):
    processed_summary = tfx.process_string(summary_string)
    if body is not None and isinstance(body, str):
        title += ' ' + body
    processed_title = tfx.process_string(title)
    yes_count_top_words = 0
    no_count_top_words = 0
    overlap_with_title = 0
    no_overlap_title = 0
    for summary_word in processed_summary:
        if pfx.TOP_POST_WORDS_100.__contains__(summary_word):
            yes_count_top_words += 1
        else:
            no_count_top_words += 1
        if processed_title.__contains__(summary_word):
            overlap_with_title += 1
        else:
            no_overlap_title += 1
    print(
        '{}% of words in preprocessed summary are from the top 100 words in the Reddit post'.format(
            yes_count_top_words / (no_count_top_words + yes_count_top_words)
        )
    )
    recall = overlap_with_title / len(processed_title)
    precision = overlap_with_title / len(processed_summary)
    print(
        'RECALL:{}/{}, {}\n'
        '     Number of overlapping words between generated summary and reference divided by\n'
        '     Total number of words in reference summary'.format(
            overlap_with_title,
            len(processed_title),
            recall
        )
    )
    print(
        'PRECISION:{}/{}, {}\n'
        '     Number of overlapping words between generated summary and reference divided by\n'
        '     Total number of words in generated summary'.format(
            overlap_with_title,
            len(processed_summary),
            precision
        )
    )
    if precision+recall != 0:
        print('F1: {}'.format((2*precision*recall)/(precision+recall)))


def calculate_rouge(hypotheses, references):
    rouge = Rouge()
    scores = rouge.get_scores(hypotheses, references, avg=True)
    return scores


pfx.COMMENT_DEPTH_FILTER = 4
pfx.ANTI_SKEW_UPV0TE_FILTER = 15
reddit_post, reddit_thread = pfx.create_dataset(science[3])
pfx.comment_structure_manipulation(reddit_post, reddit_thread, embeddings=False)
entire_post_string = tfx.string_list_to_string(pfx.FILT_ALL_THREADS_PARAGRAPHS, '', end_append=False)
entire_summary = pfx.generate_summary(entire_post_string)
print('\nPost Title\n{}'.format(reddit_post.title))
print('\nEntire Summary\n(Summary of all filtered data concatenated to one string)\n{}'.format(entire_summary))
evaluate(entire_summary, reddit_post.title)
rouge_scores = calculate_rouge(entire_summary, reddit_post.title)
print(rouge_scores)

paragraph_summaries = []
for index, paragraph in enumerate(pfx.FILT_ALL_THREADS_PARAGRAPHS):
    summary = pfx.generate_summary(paragraph)
    paragraph_summaries.append(summary)
    print('\nSummary {}\n{}'.format(index, paragraph_summaries[-1]))
main_thread_summaries = tfx.string_list_to_string(paragraph_summaries, '')
summary_of_summaries = pfx.generate_summary(main_thread_summaries)
print('\nSummary of Main Thread Summaries\n{}'.format(summary_of_summaries))
evaluate(summary_of_summaries, reddit_post.title)
