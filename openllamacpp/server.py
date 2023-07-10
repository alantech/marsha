#!/usr/bin/env python

'''
Simple and functional REST server for Python (3.5) using no dependencies beyond the Python standard library.
Based on and simplified from https://gist.github.com/iaverin/f81720df9ed37a49ecee6341e4d5c0c6
'''

import http.server
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse 
import urllib.request

# Fix issues with decoding HTTP responses
# importlib.reload(sys)

def service_worker():
    pass

def query_openllamacpp(handler):
    payload = handler.get_payload()
    result = subprocess.run(['./main', '-p', payload['query']], capture_output=True, encoding='utf8')
    return {
            "response": result.stdout,
            "stderr": result.stderr,
    }


routes = {
    r'^/$': {'POST': query_openllamacpp, 'media_type': 'application/json'}}

poll_interval = 0.1

def rest_call_json(url, payload=None, with_payload_method='PUT'):
    'REST call with JSON decoding of the response and JSON payloads'
    if payload:
        if not isinstance(payload, str):
            payload = json.dumps(payload)
        # PUT or POST
        response = urllib.request.urlopen(
            MethodRequest(url, payload.encode(), {'Content-Type': 'application/json'}, method=with_payload_method))
    else:
        # GET
        response = urllib.request.urlopen(url)
    response = response.read().decode()
    return json.loads(response)


class MethodRequest(urllib.request.Request):
    'See: https://gist.github.com/logic/2715756'

    def __init__(self, *args, **kwargs):
        if 'method' in kwargs:
            self._method = kwargs['method']
            del kwargs['method']
        else:
            self._method = None
        return urllib.request.Request.__init__(self, *args, **kwargs)

    def get_method(self, *args, **kwargs):
        return self._method if self._method is not None else urllib.request.get_method(self, *args, **kwargs)


class RESTRequestHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.routes = routes

        return http.server.BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

    def do_HEAD(self):
        self.handle_method('HEAD')

    def do_GET(self):
        self.send_response(404)
        self.end_headers()
        self.wfile.write('Route not found\n'.encode())

    def do_POST(self):
        self.handle_method('POST')

    def do_PUT(self):
        self.send_response(404)
        self.end_headers()
        self.wfile.write('Route not found\n'.encode())

    def do_DELETE(self):
        self.send_response(404)
        self.end_headers()
        self.wfile.write('Route not found\n'.encode())

    def get_payload(self):
        payload_len = int(self.headers.get('content-length', 0))
        payload = self.rfile.read(payload_len)
        payload = json.loads(payload.decode())
        return payload

    def handle_method(self, method):
        route = self.get_route()
        if route is None:
            self.send_response(404)
            self.end_headers()
            self.wfile.write('Route not found\n'.encode())
        else:
            if method == 'HEAD':
                self.send_response(200)
                if 'media_type' in route:
                    self.send_header('Content-type', route['media_type'])
                self.end_headers()
            else:
                if method in route:
                    content = route[method](self)
                    if content is not None:
                        self.send_response(200)
                        if 'media_type' in route:
                            self.send_header('Content-type', route['media_type'])
                        self.end_headers()
                        if method != 'DELETE':
                            self.wfile.write(json.dumps(content).encode())
                    else:
                        self.send_response(404)
                        self.end_headers()
                        self.wfile.write('Not found\n'.encode())
                else:
                    self.send_response(405)
                    self.end_headers()
                    self.wfile.write(method + ' is not supported\n'.encode())

    def get_route(self):
        for path, route in self.routes.items():
            if re.match(path, self.path):
                return route
        return None


def rest_server(port):
    'Starts the REST server'
    http_server = http.server.HTTPServer(('', port), RESTRequestHandler)
    http_server.service_actions = service_worker
    print('Starting HTTP server at port %d' % port)
    try:
        http_server.serve_forever(poll_interval)
    except KeyboardInterrupt:
        pass
    print('Stopping HTTP server')
    http_server.server_close()


def main(argv):
    rest_server(8765)


if __name__ == '__main__':
    main(sys.argv[1:])