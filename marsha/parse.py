from __future__ import annotations

import os

from mistletoe.block_token import Document

from marsha.ast_nodes import child_text
from marsha.meta import MarshaMeta, to_markdown, _get_ast
from marsha.utils import write_file


def split_preamble(doc: str, header: str) -> tuple[str, str]:
    # An editor response is a markdown preamble followed by the fenced artifact. Slice off the
    # preamble at the artifact's first header so the existing validators run on the artifact alone.
    marker = f'# {header}\n'
    idx = doc.find(marker)
    if idx == -1:
        raise Exception(
            f'Artifact header "# {header}" not found in editor response')
    return doc[:idx].strip(), doc[idx:]


def format_marsha_for_llm(meta: MarshaMeta) -> str:
    break_line = '\n'
    res: list[str] = [f'# Requirements for file `{meta.filename}`']
    for func in meta.functions + meta.void_funcs:
        ast = _get_ast(Document(func))
        if ast['children'][0]['type'] != 'Heading':
            raise Exception('Invalid Marsha function')
        name = ''
        args: list[str] = []
        ret = ''
        desc_parts: list[str] = []
        reqs = ''
        children = ast['children']
        last_list_i = -1
        for (i, child) in enumerate(children):
            if child['type'] == 'List':
                last_list_i = i
        for (i, child) in enumerate(children):
            if i == 0:
                # Special handling for the initial header (for now)
                if child['type'] != 'Heading':
                    raise Exception('Invalid Marsha function')
                header: str = child_text(child['children'])
                name = header.split('(')[0].split('func')[1].strip()
                args = [arg.strip()
                        for arg in header.split('(')[1].split(')')[0].split(',')]
                end = header.split('):')
                if len(end) == 1:
                    ret = 'None'
                else:
                    ret = header.split('):')[1].strip()
                continue
            if i == last_list_i:
                # The usage examples: the section's final list block (validate_marsha_fn
                # requires a section to end with its examples list); an earlier list (e.g.
                # under a "##" subsection) is description content.
                reqs = to_markdown(child)
                continue
            if last_list_i != -1 and i > last_list_i:
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


def write_files_from_markdown(md: str, subdir: str | None = None) -> list[str]:
    ast = _get_ast(Document(md))
    filenames: list[str] = []
    filename = ''
    filedata: str = ''
    for section in ast['children']:
        if section['type'] == 'Heading':
            filename = child_text(section['children'])
            if subdir is not None:
                filename = f'{subdir}/{filename}'
            filenames.append(filename)
        elif section['type'] == 'CodeFence':
            filedata = child_text(section['children'])
            if filedata == '':
                # If theres not data and we are not going to write the file, we should remove it from the filenames list
                filenames.pop()
                continue
            if subdir is not None:
                os.makedirs(os.path.dirname(filename), exist_ok=True)
            write_file(filename, filedata)
    return filenames
