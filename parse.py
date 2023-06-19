import re

from mistletoe import Document, ast_renderer


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


def format_func_for_llm(func, defined_classes: list = None):
    break_line = '\n'
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

    arg_fmt = '\n'.join([f'{i + 1}. {arg}' for (i, arg) in enumerate(args)])

    return f'''# Requirements for function `{name}`

## Inputs

{arg_fmt}

## Output

{ret}

## Description

{desc}

{defined_classes is not None and len(defined_classes) > 0 and f"""## Defined classes
{break_line.join(defined_classes)}""" or ""
}

## Examples of expected behavior

{reqs}'''


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


def extract_functions_and_types(file: str) -> tuple[list[str], list[str]]:
    res = ([], [])
    sections = file.split('#')
    func_regex = r'\s*func [a-zA-Z_][a-zA-Z0-9_]*\('
    type_regex = r'\s*type [a-zA-Z_][a-zA-Z0-9_]*'
    for section in sections:
        if re.match(func_regex, section):
            res[0].append(f'# {section.lstrip()}')
        elif re.match(type_regex, section):
            res[1].append(f'# {section.lstrip()}')
    return res


def extract_type_name(type):
    ast = ast_renderer.get_ast(Document(type))
    if ast['children'][0]['type'] != 'Heading':
        raise Exception('Invalid Marsha type')
    header = ast['children'][0]['children'][0]['content']
    return header.split('(')[0].split('type')[1].strip()


def validate_type_markdown(md, type_name):
    ast = ast_renderer.get_ast(Document(md))
    if len(ast['children']) != 2:
        return False
    if ast['children'][0]['type'] != 'Heading':
        return False
    if ast['children'][1]['type'] != 'CodeFence':
        return False
    if ast['children'][0]['children'][0]['content'].strip().lower() != f'type {type_name}'.lower():
        return False
    return True


def extract_class_definition(md):
    ast = ast_renderer.get_ast(Document(md))
    return ast['children'][1]['children'][0]['content'].strip()
