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


def read_file_list(file_path, name_list, encoding):
    """
    reads stl file and stores to array
    :param file_path: file path of ast or stl file
    :param name_list: list of file names to be appended to file_path
    :param encoding: type of encoding to use when opening file
    """
    print('Reading {} txt files ...'.format(len(name_list)))
    text_data = []
    for name in name_list:
        read_file = []
        opened_file = open(file_path + name, 'r', encoding=encoding)
        for line in opened_file:
            read_file.append(line)
        opened_file.close()
        text_data.append(read_file)
    return text_data


def load_data(src_string, encoding='cp1252'):
    text_dir = PWD + '/' + src_string + '/'
    file_names = file_search(search_term='.txt', dir_string=text_dir)
    return read_file_list(text_dir, file_names, encoding=encoding)


def save_data(data_list, dst_string):
    with open(dst_string, 'w') as file:
        for line in data_list:
            line_string = ''
            for item in line:
                line_string += item + ' '
            file.write(line_string + '\n')
