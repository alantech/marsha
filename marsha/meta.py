import os

from marsha.parse import extract_functions_and_types, extract_type_name, is_defined_from_file, extract_type_filename
from marsha.utils import read_file, get_filename_from_path


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
