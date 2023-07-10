import os
import subprocess
import tempfile
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install


script_directory = os.path.dirname(os.path.abspath(__file__))


def setup_llamacpp():
    if not os.path.exists(os.path.join(script_directory, './marsha/bin')):
        os.makedirs(os.path.join(script_directory, './marsha/bin'))
    if not os.path.exists(os.path.join(script_directory, './marsha/bin/llamacpp')):
        tmpdir = tempfile.TemporaryDirectory(
            suffix='__marsha_setup__')
        subprocess.Popen(['git', 'clone', 'https://github.com/ggerganov/llama.cpp.git'], cwd=tmpdir)
        cuda_support = True if len(subprocess.run(['command', '-v', 'nvcc'], capture_output=True, encoding='utf8').stdout) > 0 else False
        if cuda_support:
            subprocess.Popen(['make', 'LLAMA_CUBLAS=1'], cwd=os.path.join(tmpdir, './llama.cpp'))
        else:
            subprocess.Popen(['make'], cwd=os.path.join(tmpdir, './llama.cpp'))
        subprocess.Popen(['cp', os.path.join(tmpdir, './llama.cpp/main'), os.path.join(script_directory, './marsha/bin/llamacpp')])


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        setup_llamacpp()


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        setup_llamacpp()


setup(
    name='marsha',
    version='0.0.1',
    description='Marsha is a higher-level programming language.',
    url='https://github.com/alantech/marsha',
    author='Alan Technologies Maintainers',
    author_email='hello@alantechnologies.com',
    license='AGPL-3.0-only',
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
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers'
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.10',
    ],
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
)
