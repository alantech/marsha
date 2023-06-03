from urllib.parse import urlparse, parse_qs


def extract_connection_info(url):
    if url == '':
        return {}
    if url.startswith('jdbc:'):
        url = url.split('jdbc:')[1]

    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    extras = {}
    if parsed.scheme == 'postgresql' and query.get('sslmode', None):
        extras['ssl'] = query['sslmode'][0]
    if parsed.scheme == 'mysql' and query.get('sslMode', None):
        extras['ssl'] = query['sslMode'][0].lower().split('d')[0]

    dbUser = None
    if parsed.username:
        dbUser = parsed.username
    elif query.get('user', None):
        dbUser = query['user'][0]

    dbPassword = None
    if parsed.password:
        dbPassword = parsed.password
    elif query.get('password', None):
        dbPassword = query['password'][0]

    connection = {
        'protocol': parsed.scheme,
        'dbUser': dbUser,
        'dbPassword': dbPassword,
        'host': parsed.hostname,
        'port': parsed.port,
        'database': parsed.path[1:]
    }

    if extras:
        connection['extra'] = extras

    return connection
