import re
import json


def extract_connection_info(database_url):
    result = {}

    if database_url == '':
        return result

    host_database = re.compile(r'^(\w+):\/\/(\w+):(\w+)@([^\/]+)\/(\w+)$')
    match = host_database.match(database_url)

    protocol = match.group(1)
    db_user = match.group(2)
    db_password = match.group(3)
    host = match.group(4)
    database = match.group(5)

    result['protocol'] = protocol
    result['dbUser'] = db_user
    result['dbPassword'] = db_password
    result['host'] = host
    result['database'] = database

    extra = re.compile(r'^\w+:\/\/\w+:\w+@\w+\/\w+\?([\w\=&]+)$').search(database_url)
    extra_result = {}
    if extra:
        ssl_mode = re.compile(r'sslmode=([\w-]+)').search(extra.group(1))
        if ssl_mode:
            extra_result['ssl'] = ssl_mode.group(1)

        result['extra'] = extra_result

    return json.dumps(result)
