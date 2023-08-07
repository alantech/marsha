import os
import subprocess
import tempfile
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install


def setup_llamacpp(install_libbase):
    if not os.path.exists(os.path.join(install_libbase, 'marsha/bin')):
        os.makedirs(os.path.join(install_libbase, 'marsha/bin'))
    if not os.path.exists(os.path.join(install_libbase, 'marsha/bin/llamacpp')):
        with tempfile.TemporaryDirectory(
                suffix='__marsha_setup__') as tmpdir:
            print(tmpdir)
            print(subprocess.run(['bash', '-c', f'cd {tmpdir}; git clone https://github.com/ggerganov/llama.cpp.git']))
            cuda_support = True if len(subprocess.run(['bash', '-c', 'command -v nvcc'], capture_output=True, encoding='utf8').stdout) > 0 else False
            if cuda_support:
                print(subprocess.run(['bash', '-c', f'cd {os.path.join(tmpdir, "llama.cpp")}; make LLAMA_CUBLAS=1']))
            else:
                print(subprocess.run(['bash', '-c', f'cd {os.path.join(tmpdir, "llama.cpp")}; make']))
            subprocess.run(['cp', os.path.join(tmpdir, 'llama.cpp/main'), os.path.join(install_libbase, 'marsha/bin/llamacpp')])


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        setup_llamacpp()


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        setup_llamacpp(self.install_libbase)


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
