from urllib.parse import urlparse, parse_qs
import json


def extract_connection_info(url):
    if url == '':
        return {}

    parsed = urlparse(url)

    extras = {}
    if parsed.scheme == 'postgresql' and parsed.query != '':
        extras['ssl'] = parse_qs(parsed.query)['sslmode'][0]

    connection = {
        'protocol': parsed.scheme,
        'dbUser': parsed.username,
        'dbPassword': parsed.password,
        'host': parsed.hostname,
        'port': parsed.port,
        'database': parsed.path[1:]
    }

    if extras:
        connection['extra'] = extras

    return json.dumps(connection)
