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
    import inspect
    import argparse
    lookup = globals()
    func_names = [r.__name__ for r in lookup.values() if callable(r)]
    default_func = func_names[-1]
    parser = argparse.ArgumentParser(description='Marsha-generated CLI options')
    parser.add_argument('--func', action='store', required=False, choices=func_names, default=default_func)
    parser.add_argument('--stdin', action='store_true', required=False)
    parser.add_argument('--infile', action='store', required=False, default=None)
    parser.add_argument('--outfile', action='store', required=False, default=None)
    parser.add_argument('--serve', action='store', required=False, type=int)
    parser.add_argument('params', nargs='*')
    args = parser.parse_args()
    func = lookup[args.func]
    if args.serve is not None:
        from http.server import BaseHTTPRequestHandler, HTTPServer
        import json
        class MarshaServer(BaseHTTPRequestHandler):
            def do_POST(self):
                func_name = self.path.split('/')[1]
                if not func_name in func_names:
                    self.send_response(404)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(bytes('{"error": "' + self.path + ' does not exist"}', 'utf-8'))
                    return
                content_len = int(self.headers.get('Content-Length'))
                post_body = self.rfile.read(content_len)
                post_payload = None
                try:
                    post_payload = json.loads(post_body)
                except:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(bytes('{"error": "Invalid JSON provided"}', 'utf-8'))
                    return
                out = None
                if type(post_payload) is list:
                    out = func(*post_payload)
                else:
                    out = func(post_payload)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(bytes(json.dumps(out), 'utf-8'))
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
        if args.stdin:
            import sys
            param = sys.stdin.read()
            out = func(param)
        elif args.infile is not None:
            file = open(args.infile, 'r')
            param = file.read()
            file.close()
            out = func(param)
        else:
            out = func(*args.params)
        if args.outfile is not None:
            file = open(args.outfile, 'w')
            file.write(out)
            file.close()
        else:
            print(out)
""")
    f.close()
