import os

from mistletoe import Document, ast_renderer

from marsha.meta import MarshaMeta, to_markdown
from marsha.utils import write_file


def format_marsha_for_llm(meta: MarshaMeta):
    break_line = '\n'
    res = [f'# Requirements for file `{meta.filename}`']
    for func in meta.functions + meta.void_funcs:
        ast = ast_renderer.get_ast(Document(func))
        if ast['children'][0]['type'] != 'Heading':
            raise Exception('Invalid Marsha function')
        name = ''
        args = []
        ret = ''
        desc_parts = []
        reqs = ''
        list_started = False
        for (i, child) in enumerate(ast['children']):
            if i == 0:
                # Special handling for the initial header (for now)
                if child['type'] != 'Heading':
                    raise Exception('Invalid Marsha function')
                header = child['children'][0]['content']
                name = header.split('(')[0].split('func')[1].strip()
                args = [arg.strip()
                        for arg in header.split('(')[1].split(')')[0].split(',')]
                end = header.split('):')
                if len(end) == 1:
                    ret = 'None'
                else:
                    ret = header.split('):')[1].strip()
                continue
            if child['type'] == 'List':
                list_started = True
                reqs = to_markdown(child)
                continue
            if list_started:
                raise Exception(
                    'Function description must come *before* usage examples')
            desc_parts.append(to_markdown(child))
        desc = '\n\n'.join(desc_parts)

        arg_fmt = '\n'.join(
            [f'{i + 1}. {arg}' for (i, arg) in enumerate(args)])

        fn_def = f'''## Requirements for function `{name}`

### Inputs

{arg_fmt}

### Output

{ret}

### Description

{desc}

{f"""### Examples of expected behavior

{reqs}""" if len(reqs) > 0 else ''}
'''
        res.append(fn_def)
    if meta.types is not None:
        res.append('## Convert the following type into classes')
        for defined_type in meta.types:
            type_def = f'''
##{defined_type}
'''
            res.append(type_def)
    return break_line.join(res)


# TODO: Potentially re-org this so the stages are together?
def validate_first_stage_markdown(md, marsha_filename):
    ast = ast_renderer.get_ast(Document(md))
    if len(ast['children']) != 4 and len(ast['children']) != 6:
        return False
    if len(ast['children']) == 4:
        if ast['children'][0]['type'] != 'Heading':
            return False
        if ast['children'][2]['type'] != 'Heading':
            return False
        if ast['children'][1]['type'] != 'CodeFence':
            return False
        if ast['children'][3]['type'] != 'CodeFence':
            return False
        if ast['children'][0]['children'][0]['content'].strip() != f'{marsha_filename}.py':
            return False
        if ast['children'][2]['children'][0]['content'].strip() != f'{marsha_filename}_test.py':
            return False
    else:
        if ast['children'][0]['type'] != 'Heading':
            return False
        if ast['children'][2]['type'] != 'Heading':
            return False
        if ast['children'][4]['type'] != 'Heading':
            return False
        if ast['children'][1]['type'] != 'CodeFence':
            return False
        if ast['children'][3]['type'] != 'CodeFence':
            return False
        if ast['children'][5]['type'] != 'CodeFence':
            return False
        if ast['children'][0]['children'][0]['content'].strip() != f'{marsha_filename}.py':
            return False
        if ast['children'][2]['children'][0]['content'].strip() != 'requirements.txt':
            return False
        if ast['children'][4]['children'][0]['content'].strip() != f'{marsha_filename}_test.py':
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


def write_files_from_markdown(md: str, subdir=None) -> list[str]:
    ast = ast_renderer.get_ast(Document(md))
    filenames = []
    filename = ''
    filedata = ''
    for section in ast['children']:
        if section['type'] == 'Heading':
            filename = section['children'][0]['content']
            if subdir is not None:
                filename = f'{subdir}/{filename}'
            filenames.append(filename)
        elif section['type'] == 'CodeFence':
            filedata = section['children'][0]['content']
            if filedata is None or filedata == '':
                # If theres not data and we are not going to write the file, we should remove it from the filenames list
                filenames.pop()
                continue
            if subdir is not None:
                os.makedirs(os.path.dirname(filename), exist_ok=True)
            write_file(filename, filedata)
    return filenames


def extract_func_name(type) -> str:
    ast = ast_renderer.get_ast(Document(type))
    if ast['children'][0]['type'] != 'Heading':
        raise Exception('Invalid Marsha function')
    header = ast['children'][0]['children'][0]['content']
    return header.split('(')[0].split('func')[1].strip()
