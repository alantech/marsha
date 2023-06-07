from mistletoe import Document, ast_renderer


def format_func_for_llm(func):
    ast = ast_renderer.get_ast(Document(func))
    if ast['children'][0]['type'] != 'Heading':
        raise Exception('Invalid Marsha function')
    header = ast['children'][0]['children'][0]['content']
    name = header.split('(')[0].split('func')[1].strip()
    args = [arg.strip() for arg in header.split('(')[1].split(')')[0].split(',')]
    ret = header.split('):')[1].strip()
    desc_parts = []
    reqs = []
    for child in ast['children']:
        if child['type'] == 'Heading':
            continue
        if child['type'] == 'Paragraph':
            # TODO: Reconstitute *all* of the markdown formatting allowed within this section
            pg_parts = []
            for subchild in child['children']:
                if subchild['type'] == 'RawText':
                    pg_parts.append(subchild['content'])
                elif subchild['type'] == 'Emphasis':
                    pg_parts.append(f'''*{subchild['children'][0]['content']}*''')
                else:
                    continue
            desc_parts.append(''.join(pg_parts))
        if child['type'] == 'List':
            # TODO: The same "reconstitute logic should be used for each ListItem, too.
            # This could blow up in our faces easily here
            for subchild in child['children']:
                reqs.append(subchild['children'][0]['children'][0]['content'])
    desc = '\n\n'.join(desc_parts)

    arg_fmt = '\n'.join([f'{i + 1}. {arg}' for (i, arg) in enumerate(args)])
    req_fmt = '\n'.join([f'* {req}' for req in reqs])

    return f'''
# Requirements for function `{name}`

## Inputs

{arg_fmt}

## Output

{ret}

## Description

{desc}

## Examples of expected behavior

{req_fmt}'''


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
