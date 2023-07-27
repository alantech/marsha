import os
import re

from mistletoe import Document, ast_renderer

from marsha.utils import write_file


def to_markdown(node):
    # Technically I should iterate on the `children` lists every time because they could have more
    # than one, but since this is hardwired for each node type, I'm just going to use the actual
    # implementations to skip that when possible to reduce recursion depth and simplify the code
    if node['type'] == 'AutoLink':
        return f'''[{node['children'][0]['content']}]'''
    if node['type'] == 'BlockCode':
        return '\n'.join([f'''    {line}''' for line in node['children'][0].split('\n')])
    if node['type'] == 'CodeFence':
        return f'''```{node['language']}
{node['children'][0]['content']}
```'''
    if node['type'] == 'Document':
        return ''.join([to_markdown(child) for child in node['children']])
    if node['type'] == 'Emphasis':
        return f'''*{node['children'][0]['content']}*'''
    if node['type'] == 'EscapeSequence':
        return f'''\\{node['children'][0]['content']}'''
    if node['type'] == 'Heading':
        return ('#' * node['level']) + ' ' + ''.join([to_markdown(child) for child in node['children']])
    if node['type'] == 'Image':
        if len(node['title']['children'][0]['content']) > 0:
            return f'''![{''.join([to_markdown(child) for child in node['children']])}]({node['src']['children'][0]['content']} "{node['title']['children'][0]['content']}")'''
        else:
            return f'''![{''.join([to_markdown(child) for child in node['children']])}]({node['src']['children'][0]['content']})'''
    if node['type'] == 'InlineCode':
        return f'''`{node['children'][0]['content']}`'''
    if node['type'] == 'LineBreak':
        return '\n'
    if node['type'] == 'Link':
        if len(node['title']['children'][0]['content']) > 0:
            return f'''[{''.join([to_markdown(child) for child in node['children']])}]({node['src']['children'][0]['content']} "{node['title']['children'][0]['content']}")'''
        else:
            return f'''[{''.join([to_markdown(child) for child in node['children']])}]({node['src']['children'][0]['content']})'''
    if node['type'] == 'List':
        if node['start'] is not None:
            return '\n'.join([f'''{i}. {text}''' for (i, text) in enumerate([to_markdown(child) for child in node['children']])])
        else:
            return '\n'.join([f'''* {to_markdown(child)}''' for child in node['children']])
    if node['type'] == 'ListItem':
        return ''.join([to_markdown(child) for child in node['children']])
    if node['type'] == 'Paragraph':
        return ''.join([to_markdown(child) for child in node['children']])
    if node['type'] == 'Quote':
        return '\n'.join([f'''> {to_markdown(child)}''' for child in node['children']])
    if node['type'] == 'RawText':
        return node['content']
    if node['type'] == 'SetextHeading':
        raise NotImplementedError()
    if node['type'] == 'Strikethrough':
        return f'''~~{node['children'][0]['content']}~~'''
    if node['type'] == 'Strong':
        return f'''**{node['children'][0]['content']}**'''
    if node['type'] == 'Table':
        raise NotImplementedError()
    if node['type'] == 'TableCell':
        raise NotImplementedError()
    if node['type'] == 'TableRow':
        raise NotImplementedError()
    if node['type'] == 'ThematicBreak':
        return '\n---\n'
    raise Exception(f'''Unknown AST node {node['type']} encountered!''')


def format_marsha_for_llm(marsha_filename: str, functions: list[str], defined_types: list[str] = None):
    break_line = '\n'
    res = [f'# Requirements for file `{marsha_filename}`']
    for func in functions:
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
    if defined_types is not None:
        res.append(f'## Convert the following type into classes')
        for defined_type in defined_types:
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
        if ast['children'][2]['children'][0]['content'].strip() != f'requirements.txt':
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


def extract_functions_and_types(file: str) -> tuple[list[str], list[str], list[str]]:
    res = ([], [], [])
    sections = file.split('#')
    func_regex = r'\s*func [a-zA-Z_][a-zA-Z0-9_]*\(.*\):'
    void_func_regex = r'\s*func [a-zA-Z_][a-zA-Z0-9_]*\(.*\)'
    type_regex = r'\s*type [a-zA-Z_][a-zA-Z0-9_]*\s*[a-zA-Z0-9_\.\/]*'
    for section in sections:
        if re.match(void_func_regex, section) and not re.match(func_regex, section):
            void_func_str = f'# {section.lstrip()}'
            validate_marsha_fn(void_func_str, True)
            res[2].append(void_func_str)
        elif re.match(func_regex, section):
            func_str = f'# {section.lstrip()}'
            validate_marsha_fn(func_str)
            res[0].append(func_str)
        elif re.match(type_regex, section):
            type_str = f'# {section.lstrip()}'
            validate_marsha_type(type_str)
            res[1].append(type_str)
    if len(res[0]) == 0 and len(res[2]) == 0 and len(res[1]) == 0:
        raise Exception('No functions or types found in file')
    return res


def extract_type_name(type):
    ast = ast_renderer.get_ast(Document(type))
    if ast['children'][0]['type'] != 'Heading':
        raise Exception('Invalid Marsha type')
    header = ast['children'][0]['children'][0]['content']
    return header.split(' ')[1].strip()


def is_defined_from_file(md):
    ast = ast_renderer.get_ast(Document(md))
    if len(ast['children']) != 1:
        return False
    if ast['children'][0]['type'] != 'Heading':
        return False
    header = ast['children'][0]['children'][0]['content']
    split_header = header.split(' ')
    if len(split_header) != 3:
        return False
    return True


def extract_type_filename(md):
    ast = ast_renderer.get_ast(Document(md))
    header = ast['children'][0]['children'][0]['content']
    return header.split(' ')[2]


def extract_func_name(type) -> str:
    ast = ast_renderer.get_ast(Document(type))
    if ast['children'][0]['type'] != 'Heading':
        raise Exception('Invalid Marsha function')
    header = ast['children'][0]['children'][0]['content']
    return header.split('(')[0].split('func')[1].strip()


def validate_marsha_fn(fn: str, void: bool = False):
    ast = ast_renderer.get_ast(Document(fn))
    fn_heading = ast['children'][0]['children'][0]['content']
    # Check function signature
    if not void:
        return_type = fn_heading.split('):')[1].strip()
        if not return_type or return_type is None or return_type == '':
            raise Exception(
                f'Invalid Marsha function: Missing return type for `{fn_heading}`.')
    # Check description
    if ast['children'][1]['type'] != 'Paragraph':
        raise Exception(
            f'Invalid Marsha function: Invalid description for `{fn_heading}`.')
    # Extract content from all children and nested children
    fn_desc = ''
    for child in ast['children'][1]['children']:
        fn_desc += to_markdown(child)
    if len(fn_desc) <= 140:  # should not be smaller than a tweet
        raise Exception(
            f'Invalid Marsha function: Description for `{fn_heading}` is too short.')
    # Check usage examples if not void
    if not void:
        if ast['children'][2]['type'] != 'List':
            raise Exception(
                f'Invalid Marsha function: Invalid usage examples for `{fn_heading}`.')
        if len(ast['children'][2]['children']) <= 2:  # We need at least a couple of examples
            raise Exception(
                f'Invalid Marsha function: Not enough usage examples for `{fn_heading}`.')


def validate_marsha_type(type: str):
    ast = ast_renderer.get_ast(Document(type))

    if len(ast['children']) == 1:
        type_heading = ast['children'][0]['children'][0]['content']
        if len(type_heading.split(' ')) != 3:
            raise Exception(
                f'Invalid Marsha type: Invalid type definition for `{type_heading}`.')
    else:
        type_heading = ast['children'][0]['children'][0]['content']
        if ast['children'][1]['type'] != 'Paragraph':
            raise Exception(
                f'Invalid Marsha type: Invalid type definition for `{type_heading}`.')
        type_def_samples = filter(lambda x: x['type'] ==
                                  'RawText', ast['children'][1]['children'])
        if len(list(type_def_samples)) <= 2:  # We need at least the headers and a couple of examples
            raise Exception(
                f'Invalid Marsha type: Not enough examples for `{type_heading}`.')
