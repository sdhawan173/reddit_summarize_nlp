import os

PWD = os.getcwd()


def file_search(search_term, dir_string=os.getcwd(), match_term=False):
    """
    searches a directory, with the current working directory as default, for a given filetype.
    :param dir_string: string of directory to search
    :param search_term: string of filetype, input as a string in the format: '.type'
    :param match_term: Boolean to search for the exact term instead of partial term
    :return: list of file names with extensions that match search term
    """
    print('Searching for \'{}\' files ...'.format(search_term))
    file_list = []
    # Run through list and add files with .ast extension to ast_list
    for list_item in os.listdir(dir_string):
        if not match_term and list_item.__contains__(search_term):
            file_list.append(list_item)
        elif match_term and list_item == search_term:
            file_list.append(list_item)
    return sorted(file_list, key=str.casefold)


def save_data(data_list, dst_string):
    with open(dst_string, 'w') as file:
        for line in data_list:
            line_string = ''
            for item in line:
                line_string += item + ' '
            file.write(line_string + '\n')
