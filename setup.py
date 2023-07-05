from setuptools import setup

setup(
    name='marsha',
    version='0.0.1',
    description='Marsha is a higher-level programming language.',
    url='https://github.com/alantech/marsha',
    author='Alan Technologies Maintainers',
    author_email='hello@alantechnologies.com',
    license='BSD 2-clause',
    packages=['marsha'],
    install_requires=[
        'autopep8',
        'flake8',
        'mccabe',
        'mistletoe',
        'openai',
        'pycodestyle',
        'pydocstyle',
        'pyflakes',
        'pyinstaller',
        'pylama'
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.10',
    ],
)
