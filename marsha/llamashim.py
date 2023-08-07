# Minimal recreation of the async `openai.ChatCompletion.acreate` API supporting only what we use.
# Maybe some day will be broken out into a general-purpose llama.cpp wrapper API
import asyncio
import multiprocessing
import os
import subprocess


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


script_directory = os.path.dirname(os.path.abspath(__file__))
llamacpp = os.path.join(script_directory, 'bin/llamacpp')
gpu_support = True if 'gpu-layers' in subprocess.run([llamacpp, '--help'], capture_output=True, encoding='utf8').stdout else False


async def run_subprocess(stream: asyncio.subprocess.Process, timeout: float = 60.0) -> tuple[str, str]:
    stdout = ''
    stderr = ''
    try:
        stdout, stderr = await asyncio.wait_for(stream.communicate(), timeout)
    except asyncio.exceptions.TimeoutError:
        try:
            stream.kill()
        except OSError:
            # Ignore 'no such process' error
            pass
        raise Exception('run_subprocess timeout...')
    except Exception as e:
        raise e
    return (stdout.decode('utf-8'), stderr.decode('utf-8'))


async def acreate(model='gpt-3.5-turbo', messages=[], name=None, temperature=1.0, top_p=None, n=1, max_tokens=float('inf')):
    fmt_messages = '\n\n'.join([f"""{message['role'].upper()}:

{message['content']}""" for message in messages])
    req = f"""This ASSISTANT obeys the SYSTEM directives to solve the problem posed by the USER. The ASSISTANT follows the output format specified by the SYSTEM exactly, as the output directly entered into a task management system, NO COMMENTARY BEFORE OR AFTER THE DEFINED OUTPUT FORMAT.

{fmt_messages}

ASSISTANT: """
    # TODO: All of the configuration options
    args = [llamacpp, '-m', os.getenv('LLAMACPP_MODEL'), '-t', str(multiprocessing.cpu_count()), '-c', '4096', '-p', req]
    if max_tokens != float('inf'):
        args.extend(['-n', str(max_tokens)])
    if gpu_support:
        args.extend(['-ngl', '43'])  # TODO: Figure out how to determine the proper number of layers here based on GPU memory size and the model chosen
    choices = []
    for i in range(n):
        stdout, stderr = await run_subprocess(await asyncio.create_subprocess_exec(*args,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE), float('inf'))
        print(stdout, stderr)
        print(stdout.split(req)[1])
        choices.append(DotDict({'message': {'content': stdout.split(req)[1]}}))
    return DotDict({'model': model, 'usage': {'prompt_tokens': 0, 'completion_tokens': 0}, 'choices': choices})
