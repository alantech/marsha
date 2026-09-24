from __future__ import annotations

import os
import re
from typing import cast

from mistletoe import ast_renderer
from mistletoe.block_token import Document

from marsha.ast_nodes import AstNode, DocumentNode, child_text
from marsha.utils import read_file, get_filename_from_path


def _get_ast(doc: Document) -> DocumentNode:
    # mistletoe's ast_renderer.get_ast is untyped; wrap it in a typed boundary that returns the
    # root markdown-AST node (a Document). The concrete node dicts carry extra keys (line
    # numbers, alignment) that marsha does not read, so they are not checked here.
    return cast('DocumentNode', ast_renderer.get_ast(doc))


def to_markdown(node: AstNode) -> str:
    # Technically I should iterate on the `children` lists every time because they could have more
    # than one, but since this is hardwired for each node type, I'm just going to use the actual
    # implementations to skip that when possible to reduce recursion depth and simplify the code
    if node['type'] == 'AutoLink':
        return f'''[{child_text(node['children'])}]'''
    if node['type'] == 'BlockCode':
        return '\n'.join([f'''    {line}''' for line in child_text(node['children']).split('\n')])
    if node['type'] == 'CodeFence':
        return f'''```{node['language']}
{child_text(node['children'])}
```'''
    if node['type'] == 'Document':
        return ''.join([to_markdown(child) for child in node['children']])
    if node['type'] == 'Emphasis':
        return f'''*{child_text(node['children'])}*'''
    if node['type'] == 'EscapeSequence':
        return f'''\\{child_text(node['children'])}'''
    if node['type'] == 'Heading':
        level: int = node['level']
        return ('#' * level) + ' ' + ''.join([to_markdown(child) for child in node['children']])
    if node['type'] == 'Image':
        # mistletoe stores an image's src and title as plain strings (empty title when absent).
        alt = ''.join([to_markdown(child) for child in node['children']])
        title = node['title']
        if len(title) > 0:
            return f'''![{alt}]({node['src']} "{title}")'''
        return f'''![{alt}]({node['src']})'''
    if node['type'] == 'InlineCode':
        return f'''`{child_text(node['children'])}`'''
    if node['type'] == 'LineBreak':
        return '\n'
    if node['type'] == 'Link':
        # mistletoe stores a link's destination as `target` (not `src`) and its title as a plain
        # string (empty when absent).
        text = ''.join([to_markdown(child) for child in node['children']])
        title = node['title']
        if len(title) > 0:
            return f'''[{text}]({node['target']} "{title}")'''
        return f'''[{text}]({node['target']})'''
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
        return f'''~~{child_text(node['children'])}~~'''
    if node['type'] == 'Strong':
        return f'''**{child_text(node['children'])}**'''
    if node['type'] == 'Table':
        raise NotImplementedError()
    if node['type'] == 'TableCell':
        raise NotImplementedError()
    if node['type'] == 'TableRow':
        raise NotImplementedError()
    if node['type'] == 'ThematicBreak':
        return '\n---\n'
    raise Exception(f'''Unknown AST node {node['type']} encountered!''')


def validate_marsha_fn(fn: str, void: bool = False) -> None:
    ast = _get_ast(Document(fn))
    first = ast['children'][0]
    if first['type'] != 'Heading':
        raise Exception('Invalid Marsha function')
    fn_heading: str = child_text(first['children'])
    # Check function signature
    if not void:
        return_type: str = fn_heading.split('):')[1].strip()
        if not return_type or return_type is None or return_type == '':
            raise Exception(
                f'Invalid Marsha function: Missing return type for `{fn_heading}`.')
    # Check description
    second = ast['children'][1]
    if second['type'] != 'Paragraph':
        raise Exception(
            f'Invalid Marsha function: Invalid description for `{fn_heading}`.')
    # Check usage examples if not void first because we need to check the length later
    if not void:
        last = ast['children'][-1]
        if last['type'] != 'List':
            raise Exception(
                f'Invalid Marsha function: Invalid usage examples for `{fn_heading}`.')
        if len(last['children']) < 2:  # We need at least a couple of examples
            raise Exception(
                f'Invalid Marsha function: Not enough usage examples for `{fn_heading}`.')
    # Extract the description (the block nodes between the header and the trailing examples list,
    # if any). to_markdown on a block node already concatenates its children, so this matches the
    # former per-child iteration.
    fn_desc = ''
    range_stop = len(ast['children']) - 1 if not void else len(ast['children'])
    for i in range(1, range_stop):
        fn_desc += to_markdown(ast['children'][i])
    if len(fn_desc) <= 80:  # around a couple of sentences at least
        raise Exception(
            f'Invalid Marsha function: Description for `{fn_heading}` is too short.')


def validate_marsha_type(type: str) -> None:
    ast = _get_ast(Document(type))
    first = ast['children'][0]
    if first['type'] != 'Heading':
        raise Exception('Invalid Marsha type')
    type_heading: str = child_text(first['children'])

    if len(ast['children']) == 1:
        if len(type_heading.split(' ')) != 3:
            raise Exception(
                f'Invalid Marsha type: Invalid type definition for `{type_heading}`.')
    else:
        second = ast['children'][1]
        if second['type'] != 'Paragraph':
            raise Exception(
                f'Invalid Marsha type: Invalid type definition for `{type_heading}`.')
        type_def_samples = filter(
            lambda x: x['type'] == 'RawText', second['children'])
        if len(list(type_def_samples)) <= 2:  # We need at least the headers and a couple of examples
            raise Exception(
                f'Invalid Marsha type: Not enough examples for `{type_heading}`.')


def extract_functions_and_types(file: str) -> tuple[list[str], list[str], list[str]]:
    res: tuple[list[str], list[str], list[str]] = ([], [], [])
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
    types_defined: list[str] = []
    for raw_type in raw_types:
        type_name = extract_type_name(raw_type)
        # If type is defined from a file, read the file
        if is_defined_from_file(raw_type):
            print('Reading type from file...')
            filename = extract_type_filename(raw_type)
            full_path = f'{dirname}/{filename}'
            try:
                type_data = cast(str, read_file(full_path))
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


def extract_type_name(type: str) -> str:
    ast = _get_ast(Document(type))
    first = ast['children'][0]
    if first['type'] != 'Heading':
        raise Exception('Invalid Marsha type')
    header: str = child_text(first['children'])
    return header.split(' ')[1].strip()


def is_defined_from_file(md: str) -> bool:
    ast = _get_ast(Document(md))
    if len(ast['children']) != 1:
        return False
    first = ast['children'][0]
    if first['type'] != 'Heading':
        return False
    header: str = child_text(first['children'])
    split_header = header.split(' ')
    if len(split_header) != 3:
        return False
    return True


def extract_type_filename(md: str) -> str:
    ast = _get_ast(Document(md))
    first = ast['children'][0]
    if first['type'] != 'Heading':
        raise Exception('Invalid Marsha type')
    header: str = child_text(first['children'])
    return header.split(' ')[2]


def extract_func_name(type: str) -> str:
    ast = _get_ast(Document(type))
    first = ast['children'][0]
    if first['type'] != 'Heading':
        raise Exception('Invalid Marsha function')
    header: str = child_text(first['children'])
    return header.split('(')[0].split('func')[1].strip()


def void_note(meta: MarshaMeta) -> str:
    # The note telling oracle-related prompts not to test the void functions (grammar-level,
    # target-language-agnostic). Empty when the assignment has no void functions.
    void_function_names: list[str] = list(
        map(lambda f: extract_func_name(f), meta.void_funcs))
    if len(void_function_names) == 0:
        return ''
    return f'Do not create any tests for the void functions: {", ".join(void_function_names)}.'


class MarshaMeta():
    input_file: str
    filename: str
    content: str
    functions: list[str]
    void_funcs: list[str]
    types: list[str] | None

    def __init__(self, input_file: str) -> None:
        self.input_file = input_file

    async def populate(self) -> MarshaMeta:
        marsha_file_dirname = os.path.dirname(self.input_file)
        self.filename = get_filename_from_path(self.input_file)
        self.content = cast(str, read_file(self.input_file))
        self.functions, types, self.void_funcs = extract_functions_and_types(
            self.content)
        self.types = None
        # Pre-process types in case we need to open a file to get the type definition
        if len(types) > 0:
            self.types = await process_types(types, marsha_file_dirname)

        return self
