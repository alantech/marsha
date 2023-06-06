from mistletoe import Document, ast_renderer


def extract_function_name(func):
    ast = ast_renderer.get_ast(Document(func))
    if ast['children'][0]['type'] != 'Heading':
        raise Exception('Invalid Marsha function')
    header = ast['children'][0]['children'][0]['content']
    return header.split('(')[0].split('func')[1].strip()


# TODO: Potentially re-org this so the stages are together?
def validate_first_stage_markdown(md, func_name):
    ast = ast_renderer.get_ast(Document(md))
    if len(ast['children']) != 4:
        return False
    if ast['children'][0]['type'] != 'Heading':
        return False
    if ast['children'][2]['type'] != 'Heading':
        return False
    if ast['children'][1]['type'] != 'CodeFence':
        return False
    if ast['children'][3]['type'] != 'CodeFence':
        return False
    if ast['children'][0]['children'][0]['content'].strip() != f'{func_name}.py':
        return False
    if ast['children'][2]['children'][0]['content'].strip() != f'{func_name}_test.py':
        return False
    return True


def validate_second_stage_markdown(md, filename):
    ast = ast_renderer.get_ast(Document(md))
    if len(ast['children']) != 2:
        return False
    if ast['children'][0]['type'] != 'Heading':
        return False
    if ast['children'][1]['type'] != 'CodeFence':
        return False
    if ast['children'][0]['children'][0]['content'].strip() != filename:
        return False
    return True


def write_files_from_markdown(md):
    ast = ast_renderer.get_ast(Document(md))
    filenames = []
    filename = ''
    filedata = ''
    for section in ast['children']:
        if section['type'] == 'Heading':
            filename = section['children'][0]['content']
            filenames.append(filename)
        elif section['type'] == 'CodeFence':
            filedata = section['children'][0]['content']
            f = open(filename, 'w')
            f.write(filedata)
            f.close()
    return filenames
