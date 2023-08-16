import os
import re

from mistletoe import Document, ast_renderer

from marsha.utils import read_file, get_filename_from_path


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
    # Check usage examples if not void first because we need to check the length later
    if not void:
        if ast['children'][-1]['type'] != 'List':
            raise Exception(
                f'Invalid Marsha function: Invalid usage examples for `{fn_heading}`.')
        if len(ast['children'][-1]['children']) < 2:  # We need at least a couple of examples
            raise Exception(
                f'Invalid Marsha function: Not enough usage examples for `{fn_heading}`.')
    # Extract content from all children and nested children except header and examples if any
    fn_desc = ''
    range_stop = len(ast['children']) - 1 if not void else len(ast['children'])
    for i in range(1, range_stop):
        for child in ast['children'][i]['children']:
            fn_desc += to_markdown(child)
    if len(fn_desc) <= 80:  # around a couple of sentences at least
        raise Exception(
            f'Invalid Marsha function: Description for `{fn_heading}` is too short.')


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


async def process_types(raw_types: list[str], dirname: str) -> list[str]:
    types_defined = []
    for raw_type in raw_types:
        type_name = extract_type_name(raw_type)
        # If type is defined from a file, read the file
        if is_defined_from_file(raw_type):
            print('Reading type from file...')
            filename = extract_type_filename(raw_type)
            full_path = f'{dirname}/{filename}'
            try:
                type_data = read_file(full_path)
            except Exception:
                err = f'Failed to read file: {full_path}'
                # if args.debug:
                #     print(err)
                #     print(e)
                raise Exception(err)
            raw_type = f'''# type {type_name}
{type_data}
            '''
        types_defined.append(raw_type)
    return types_defined


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


class MarshaMeta():
    def __init__(self, input_file):
        self.input_file = input_file

    async def populate(self):
        marsha_file_dirname = os.path.dirname(self.input_file)
        self.filename = get_filename_from_path(self.input_file)
        self.content = read_file(self.input_file)
        self.functions, types, self.void_funcs = extract_functions_and_types(
            self.content)
        self.types = None
        # Pre-process types in case we need to open a file to get the type definition
        if len(types) > 0:
            self.types = await process_types(types, marsha_file_dirname)

        return self
