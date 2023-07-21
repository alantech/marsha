import autopep8
import os
import shutil


def read_file(filename: str, mode: str = 'r'):
    f = open(filename, mode)
    content = f.read()
    f.close()
    return content


def write_file(filename: str, content: str, mode: str = 'w'):
    f = open(filename, mode)
    f.write(content)
    f.close()


def autoformat_files(files: list[str]):
    for file in files:
        before = read_file(file)
        after = autopep8.fix_code(before)
        write_file(file, after)


def copy_file(src: str, dest: str):
    shutil.copyfile(src, dest)


def delete_dir_and_content(filename: str):
    dir = os.path.dirname(filename)
    if os.path.isdir(dir):
        shutil.rmtree(dir)


def get_filename_from_path(path: str):
    return os.path.splitext(os.path.basename(path))[0]

def add_helper(filename: str):
    f = open(filename, 'a')
    f.write("""
if __name__ == '__main__':
    import argparse
    import inspect
    import json
    lookup = globals()
    func_names = [r.__name__ for r in lookup.values() if callable(r)]
    default_func = func_names[-1]
    parser = argparse.ArgumentParser(description='Marsha-generated CLI options')
    parser.add_argument('-c', '--func', action='store', required=False, choices=func_names, default=default_func,
            help='Specifies the function to call. Defaults to the last defined function')
    parser.add_argument('-j', '--force-json', action='store_true', required=False,
            help='Forces arguments, files, or stdin to be parsed as JSON')
    parser.add_argument('-t', '--force-text', action='store_true', required=False,
            help='Forces arguments, files, or stdin to be parsed as raw text')
    parser.add_argument('-i', '--stdin', action='store_true', required=False,
            help='Ignores CLI parameters in favor of stdin (as a single parameter)')
    parser.add_argument('-f', '--infile', action='store', required=False, default=None,
            help='Ignores CLI parameters in favor of reading the specified file (as a single parameter)')
    parser.add_argument('-o', '--outfile', action='store', required=False, default=None,
            help='Saves the result to a file instead of stdout')
    parser.add_argument('-s', '--serve', action='store', required=False, type=int,
            help='Spins up a simple REST web server on the specified port. When used all other options are ignored')
    parser.add_argument('params', nargs='*', help='Arguments to be provided to the function being run. Optimistically converted to simple python types by default, and left as strings if not possible')
    args = parser.parse_args()
    func = lookup[args.func]
    if args.serve is not None:
        from http.server import BaseHTTPRequestHandler, HTTPServer
        class MarshaServer(BaseHTTPRequestHandler):
            def do_POST(self):
                func_name = self.path.split('/')[1]
                if not func_name in func_names:
                    self.send_response(404)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(bytes('{"error": "' + self.path + ' does not exist"}', 'utf-8'))
                    return
                func = lookup[func_name]
                content_len = int(self.headers.get('Content-Length', 0))
                post_body = self.rfile.read(content_len)
                post_payload = None
                is_json = self.headers.get_content_type() == 'application/json'
                if is_json:
                    try:
                        post_payload = json.loads(post_body)
                    except:
                        self.send_response(400)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(bytes('{"error": "Invalid JSON provided"}', 'utf-8'))
                        return
                else:
                    post_payload = post_body.decode('utf-8')
                out = None
                try:
                    if type(post_payload) is list:
                        out = func(*post_payload)
                    else:
                        out = func(post_payload)
                    self.send_response(200)
                except Exception as e:
                    self.send_response(400)
                    if is_json:
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(bytes('{"error": "' + str(e) + '"}', 'utf-8'))
                    else:
                        self.send_header('Content-Type', 'text/plain')
                        self.end_headers()
                        self.wfile.write(bytes(str(e), 'utf-8'))
                    return
                if is_json:
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(bytes(json.dumps(out), 'utf-8'))
                else:
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(bytes(out, 'utf-8'))
        server = HTTPServer(('', args.serve), MarshaServer)
        print(f'Listening on port {args.serve}')
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass

        server.server_close()
        print("Server stopped.")
    else:
        out = None
        parsed_param = None
        as_json = False
        if args.stdin:
            import sys
            param = sys.stdin.read()
            if args.force_json:
                parsed_param = json.loads(param)
                as_json = True
            elif args.force_text:
                parsed_param = param
                as_json = False
            else:
                try:
                    parsed_param = json.loads(param)
                    as_json = True
                except:
                    parsed_param = param
                    as_json = False
        elif args.infile is not None:
            file = open(args.infile, 'r')
            param = file.read()
            file.close()
            if args.force_json:
                parsed_param = json.loads(param)
                as_json = True
            elif args.force_text:
                parsed_param = param
                as_json = False
            else:
                try:
                    parsed_param = json.loads(param)
                    as_json = True
                except:
                    parsed_param = param
                    as_json = False
        else:
            if args.force_json:
                parsed_param = [json.loads(param) for param in args.params]
                as_json = True
            elif args.force_text:
                parsed_param = args.params
                as_json = False
            else:
                try:
                    parsed_param = [json.loads(param) for param in args.params]
                    as_json = True
                except:
                    parsed_param = args.params
                    as_json = False
        if type(parsed_param) is list:
            out = func(*parsed_param)
        else:
            out = func(parsed_param)
        if args.outfile is not None:
            file = open(args.outfile, 'w')
            file.write(json.dumps(out) if as_json else out)
            file.close()
        else:
            print(json.dumps(out) if as_json else out)
""")
    f.close()
