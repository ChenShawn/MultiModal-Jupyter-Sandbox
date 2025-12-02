# saveload vars

import io
import dill
import logging
import fnmatch  # unix wildcard variable naming to save
from typing import Optional, Union  # variable typing for user hints

import logging

# general logger. Produces nice output format with live hyperlinks for pycharm users
# to use it, just call log=get_logger() at the top of your python file
# all these loggers share the same logger name 'NE1'

_LOGGING_LEVEL = logging.INFO  # usually INFO is good, DEBUG for debugging
_DEFAULT_FILENAME = 'saveloadvars-variables'
_DILL_EXTENSION = '.dill'
_RAN_SAVELOADVARS_TODAY_FILENAME = 'saveloadvars-ran.txt'


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""
    # see https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output/7995762#7995762

    # \x1b[ (ESC[) is the CSI introductory sequence for ANSI https://en.wikipedia.org/wiki/ANSI_escape_code
    # The control sequence CSI n m, named Select Graphic Rendition (SGR), sets display attributes.
    grey = "\x1b[2;37m"  # 2 faint, 37 gray
    yellow = "\x1b[33;21m"
    cyan = "\x1b[0;36m"  # 0 normal 36 cyan
    green = "\x1b[31;21m"  # dark green
    red = "\x1b[31;21m"  # bold red
    bold_red = "\x1b[31;1m"
    light_blue = "\x1b[1;36m"
    blue = "\x1b[1;34m"
    reset = "\x1b[0m"
    # File "{file}", line {max(line, 1)}'.replace("\\", "/")
    format = '[%(levelname)s]: %(asctime)s - %(name)s - %(message)s (File "%(pathname)s", line %(lineno)d, in %(funcName)s)'

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: cyan + format + reset,
        logging.WARNING: red + format + reset,
        logging.ERROR: bold_red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record).replace("\\", "/")  # replace \ with / for pycharm links


def get_logger(level=_LOGGING_LEVEL):
    """ Use get_logger to define a logger with useful color output and info and warning turned on according to the global LOGGING_LEVEL.

    :returns: the logger.
    """
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(
        'saveloadvars')  # tobi changed so all have same name so we can uniformly affect all of them
    logger.setLevel(level)
    # create console handler if this logger does not have handler yet
    if len(logger.handlers) == 0:
        ch = logging.StreamHandler()
        ch.setFormatter(CustomFormatter())
        logger.addHandler(ch)
    return logger


# https://stackoverflow.com/questions/6618795/get-locals-from-calling-namespace-in-python
import inspect


def _is_var(k, v):
    """ Central function to determine if a variable is of type that should be saved

    :param k: the key (name)
    :param v: its value
    :return: True if variable, False if not
    """
    from types import ModuleType
    isvar = k.startswith('_') \
            or (isinstance(k, str) and (k == 'tmp' or k == 'In' or k == 'Out' or k == 'monkeypatch')) \
            or hasattr(v, '__call__') \
            or isinstance(v, ModuleType) \
            or isinstance(v, logging.Logger)
    return isvar


def printvars(logging_level=_LOGGING_LEVEL):
    """ prints local variables, similar to %who in ipython jupyter notebook.

    :return: the comma-separated list of variable names as a string
    """

    locals = None
    log = get_logger(logging_level)

    frame = inspect.currentframe().f_back
    try:
        locals = frame.f_locals
    finally:
        del frame
    if locals is None:
        print('No variables found in this workspace')
        return ''
    print('Non-Jupyter-notebook variables in this workspace: ', end='')
    s = ''
    for k, v in locals.items():
        if _is_var(k, v):
            continue
        s += k + ','
    s = s[:-1]  # remove last comma
    print(s)
    return s


def dump_vars_to_bytes(vars: Union[list, str, None] = None, logging_level=_LOGGING_LEVEL):
    log = get_logger(logging_level)

    import inspect, dill
    locals = None
    frame = inspect.currentframe().f_back
    try:
        locals = frame.f_locals
    finally:
        del frame
    if locals is None:
        log.warning('found zero variables to save')
        return None

    if not vars is None:
        if not type(vars) is list and not type(vars) is str:
            raise ValueError(
                'vars argument must be a list of strings, e.g. ["a","b"], or a wildcard string, e.g. "a*", not {vars}')

        vars_to_save = dict()  # a temp dict to save the ones we really want to save
        if type(vars) is list:
            def is_list_of_strings(lst):
                return bool(lst) and not isinstance(lst, str) and all(isinstance(elem, str) for elem in lst)

            if not is_list_of_strings(vars):
                raise ValueError('vars argument is a list but it must be a list of strings')

            log.debug(f'Matching variables to save using list of variables {vars}')
            # vars is a list of string names of vars
            for v in vars:
                if not v in locals.keys():
                    raise ValueError(f"'{v}' is not a local variable")
                else:
                    vars_to_save[v] = locals[v]
            locals = vars_to_save
        elif type(vars) is str:
            log.debug(f'Matching variables to save using wildcard expression "{vars}"')
            for v in locals:
                if fnmatch.fnmatch(v, vars):
                    log.debug(f'matched local variable "v" to wildcard "{vars}"')
                    vars_to_save[v] = locals[v]
        if len(vars_to_save) == 0:
            log.warning(f'Cancelling - no local variables matched vars={vars}')
            return None
        locals = vars_to_save

    data = {}
    could_not_pickle = []

    from types import ModuleType
    s = ''
    for k, v in locals.items():
        # don't try to pickle any objects that we don't want from notebook and anything that doesn't pickle
        if _is_var(k, v):
            continue
        try:
            if not dill.pickles(v):
                could_not_pickle.append(k)
                continue
        except:
            could_not_pickle.append(k)
            continue
        s = s + k + ' '
        data[k] = v
    s = s + ']'
    if len(data) == 0:
        log.warning('Could not find any local variables to save')
        return None

    buffer = io.BytesIO()
    try:
        try:
            dill.dump(data, buffer)
            log.info(f'Saved to BytesIO variables [ {data} ]')
            if len(could_not_pickle) > 0:
                log.warning(f'could not pickle: {could_not_pickle}')
        except TypeError as e:
            log.error(f'\n Error: {e}')
    except Exception as e:
        log.error('could not save data to BytesIO')

    return buffer.getvalue()


def savevars(filename: str = _DEFAULT_FILENAME, vars: Union[list, str, None] = None,
             overwrite: Union[bool, str] = 'prompt', logging_level=_LOGGING_LEVEL):
    """
    saves all local variables to a file with dill
    
    :param filename: the name of the file. The suffix .dill is added if there is not a suffix already.
        A default filename is used with no arguments.
    :param vars: a list of string names of variables to save.
        If `vars=None` (default), all local variables are saved.
        If `vars` is a single string,
        then it is treated as a unix style wildcard expression for matching variable names
        using `fnmatch <https://docs.python.org/3/library/fnmatch.html#fnmatch.fnmatch>`_, e.g. `vars='a*`' will save
        all variables starting with `a`. If `vars` is a list of strings, then these variables are saved.
    :param overwrite: 'prompt' (default) prompts for overwrite of existing file,
        'yes' or `True` overwrites, 'no'  or `False` does not overwrite.
    :param: logging_level: set a different logging level (for testing)
    """
    log = get_logger(logging_level)

    import inspect, dill
    locals = None
    frame = inspect.currentframe().f_back
    try:
        locals = frame.f_locals
    finally:
        del frame
    if locals is None:
        log.warning('found zero variables to save')
        return
    if not vars is None:
        if not type(vars) is list and not type(vars) is str:
            raise ValueError(
                'vars argument must be a list of strings, e.g. ["a","b"], or a wildcard string, e.g. "a*", not {vars}')

        vars_to_save = dict()  # a temp dict to save the ones we really want to save
        if type(vars) is list:
            def is_list_of_strings(lst):
                return bool(lst) and not isinstance(lst, str) and all(isinstance(elem, str) for elem in lst)

            if not is_list_of_strings(vars):
                raise ValueError('vars argument is a list but it must be a list of strings')

            log.debug(f'Matching variables to save using list of variables {vars}')
            # vars is a list of string names of vars
            for v in vars:
                if not v in locals.keys():
                    raise ValueError(f"'{v}' is not a local variable")
                else:
                    vars_to_save[v] = locals[v]
            locals = vars_to_save
        elif type(vars) is str:
            log.debug(f'Matching variables to save using wildcard expression "{vars}"')
            for v in locals:
                if fnmatch.fnmatch(v, vars):
                    log.debug(f'matched local variable "v" to wildcard "{vars}"')
                    vars_to_save[v] = locals[v]
        if len(vars_to_save) == 0:
            log.warning(f'Cancelling - no local variables matched vars={vars}')
            return
        locals = vars_to_save

    data = {}
    could_not_pickle = []

    from types import ModuleType
    s = ''
    for k, v in locals.items():
        # don't try to pickle any objects that we don't want from notebook and anything that doesn't pickle
        if _is_var(k, v):
            continue
        try:
            if not dill.pickles(v):
                could_not_pickle.append(k)
                continue
        except:
            could_not_pickle.append(k)
            continue
        s = s + k + ' '
        data[k] = v
    s = s + ']'
    if len(data) == 0:
        log.warning('Could not find any local variables to save')
        return
    if filename is None:
        log.error('you must supply a filename')
        return
    from pathlib import Path
    dill_file_path = Path(filename)
    if dill_file_path.suffix == '':  # if suffix is missing add .dill
        dill_file_path = dill_file_path.parent / (dill_file_path.name + _DILL_EXTENSION)
    if not type(overwrite) is bool and not overwrite in ('yes', 'no', 'prompt'):
        raise ValueError(f"argument 'overwrite' must be True or False or one of ('yes', 'no', 'prompt')")
    if dill_file_path.exists():
        if overwrite == False or overwrite == 'no' or (
                overwrite == 'prompt' and _yes_or_no_or_always(f'file {dill_file_path} already exists, overwrite it?',
                                                               default='y', always_option=False) == 'n'):
            log.info('cancelled')
            return
    try:
        with open(dill_file_path, 'wb') as f:
            try:
                dill.dump(data, f)
                log.info(f'Saved to {dill_file_path} variables [ {s}')  # show what we saved
                if len(could_not_pickle) > 0:
                    log.warning(f'could not pickle: {could_not_pickle}')
            except TypeError as e:
                log.error(f'\n Error: {e}')
    except Exception as e:
        log.error(f'could not save data to {dill_file_path}')


def load_vars_from_bytes(bytes_data, logging_level=_LOGGING_LEVEL, overwrite: Union[str, bool] = 'yes', warn: bool = True):
    log = get_logger(logging_level)
    assert (type(overwrite) is bool or overwrite == 'yes' or overwrite == 'no' or overwrite == 'prompt'), \
        f'overwrite={overwrite} is invalid, must be bool, "yes", "no", or "prompt"'

    if not bytes_data:
        return None

    buffer = io.BytesIO(bytes_data)
    try:
        data = dill.load(buffer)
    except Exception as e:
        log.error(f'could load load dill: got {e}')
        raise (e)

    import inspect
    try:
        frame = inspect.currentframe().f_back  # get the workspace frame (jupyter workspace frame)
        locals = frame.f_locals  # get its local variable dict
        always = False
        for k in data:
            try:
                if k in locals.keys():  # if variable exists
                    if overwrite == 'yes' or always or ((type(overwrite) is bool) and overwrite):
                        # if always overwrite (yes) or prompt returns True
                        locals[k] = data[k]  # set a value in it
                        overwrote.append(k)
                        if overwrite == 'yes':
                            log.debug(f'user answered yes to overwriting variable "{k}"')
                    elif overwrite == 'prompt':
                        resp = _yes_or_no_or_always(f'Overwrite existing variable "{k}" ?')
                        if resp == 'always':
                            always = True
                            locals[k] = data[k]  # set a value in it
                            overwrote.append(k)
                        elif resp == 'yes':
                            locals[k] = data[k]  # set a value in it
                            overwrote.append(k)
                            if overwrite == 'yes':
                                log.debug(f'user answered yes to overwriting variable "{k}"')
                        else:
                            did_not_overwrite.append(k)
                    else:
                        did_not_overwrite.append(k)
                else:
                    locals[k] = data[k]  # set a value in it
            except Exception as e:
                log.error(f'could not set variable {k}: got {e}')
    finally:
        del frame


def loadvars(filename: str = _DEFAULT_FILENAME, overwrite: Union[str, bool] = 'prompt', warn: bool = True,
             logging_level=_LOGGING_LEVEL):
    """ Loads variables from file into the current workspace
    This function loads the variables found in filename into the parent workspace.
    
    :param filename: the dill file to load from, e.g. lab1. The suffix .dill is added automatically unless there is already a suffix.
        A default filename is loaded if it is not supplied.
    :param overwrite: 'prompt' (default) asks to overwrite, 'no' or `False` does not overwrite, 'yes' or `True` overwrites silently.
    :param warn: `True`, warn the user one a day about dangers of unpickling any file,
        `False`, don't warn
    :param logging_level: set a different logging level.
        default is logging.INFO. Use it for testing, must import logging and use e.g. logging.DEBUG

    :raises ExceptionType: if it cannot load for any reason except \
        not being able to overwrite particular variables.
    """
    log = get_logger(logging_level)
    assert (type(overwrite) is bool or overwrite == 'yes' or overwrite == 'no' or overwrite == 'prompt'), \
        f'overwrite={overwrite} is invalid, must be bool, "yes", "no", or "prompt"'
    import dill
    from pathlib import Path
    dill_file_path = Path(filename)
    if dill_file_path.suffix == '':  # if suffix is missing add .dill
        dill_file_path = dill_file_path.parent / (dill_file_path.name + _DILL_EXTENSION)
    assert FileNotFoundError, f'Path {dill_file_path} does not exist'
    from os import access, R_OK
    from os.path import isfile
    assert isfile(dill_file_path) and access(dill_file_path, R_OK), \
        f"File {dill_file_path} doesn't exist or isn't readable"
    if warn:
        try:
            import tempfile, os, pathlib, time
            ran_today_path = Path(os.path.join(tempfile.tempdir, _RAN_SAVELOADVARS_TODAY_FILENAME))
            warning_msg = f'Unpickling file "{dill_file_path}" can be used maliciously to execute arbitrary code.\nThis warning (shown once per 24h) can be suppressed with argument warn=False.\n Do you trust "{dill_file_path}"?'
            if ran_today_path.exists():
                seconds_since_last_ran = time.time() - os.path.getmtime(ran_today_path)
                log.debug(f'seconds since saveloadvars last ran: {seconds_since_last_ran}')
                if seconds_since_last_ran > 24 * 60 * 60:
                    confirm = _yes_or_no_or_always(warning_msg, default='n', always_option=False)
                    if not confirm == 'yes':
                        log.info('cancelled loadvars from possibly unsafe dill')
                        return
            else:
                confirm = _yes_or_no_or_always(warning_msg, default='n', always_option=False)
                if not confirm == 'yes':
                    log.info('cancelled  loadvars from possibly unsafe dill')
                    return
            ran_today_path.touch(exist_ok=True)
            log.debug(f'touched {ran_today_path}')
        except Exception as e:
            log.warning(f'could not warning user about unpickling: {e}')
    did_not_overwrite = []
    overwrote = []
    with open(dill_file_path, 'rb') as f:
        try:
            data = dill.load(f)
        except Exception as e:
            log.error(f'could load load dill: got {e}')
            raise (e)
        log.info(f'from {dill_file_path} loaded variables {list(data.keys())}')
        import inspect
        try:
            frame = inspect.currentframe().f_back  # get the workspace frame (jupyter workspace frame)
            locals = frame.f_locals  # get its local variable dict
            always = False
            for k in data:
                try:
                    if k in locals.keys():  # if variable exists
                        if overwrite == 'yes' or always or ((type(overwrite) is bool) and overwrite):
                            # if always overwrite (yes) or prompt returns True
                            locals[k] = data[k]  # set a value in it
                            overwrote.append(k)
                            if overwrite == 'yes':
                                log.debug(f'user answered yes to overwriting variable "{k}"')
                        elif overwrite == 'prompt':
                            resp = _yes_or_no_or_always(f'Overwrite existing variable "{k}" ?')
                            if resp == 'always':
                                always = True
                                locals[k] = data[k]  # set a value in it
                                overwrote.append(k)
                            elif resp == 'yes':
                                locals[k] = data[k]  # set a value in it
                                overwrote.append(k)
                                if overwrite == 'yes':
                                    log.debug(f'user answered yes to overwriting variable "{k}"')
                            else:
                                did_not_overwrite.append(k)
                        else:
                            did_not_overwrite.append(k)
                    else:
                        locals[k] = data[k]  # set a value in it
                except Exception as e:
                    log.error(f'could not set variable {k}: got {e}')
        finally:
            del frame
        # didnt_overwrite_str=' '.join(did_not_overwrite)
        if len(overwrote) > 0:
            log.info(f'overwrote existing variables {overwrote}')
        if len(did_not_overwrite) > 0:
            log.info(f'did not overwrite existing variables {did_not_overwrite}')


# useful utilities to ask question at console terminal with default answer and timeout

import signal
import time


def _alarm_handler(signum, frame):
    raise TimeoutError


def _input_with_timeout(prompt, timeout=30):
    """ get input with timeout

    :param prompt: the prompt to print
    :param timeout: timeout in seconds, or None to disable

    :returns: the input
    :raises TimeoutError: if times out
    """
    # set signal handler
    if timeout is not None:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout)  # produce SIGALRM in `timeout` seconds
    try:
        time.sleep(.5)  # get input to be printed after logging
        return input(prompt)
    except TimeoutError as to:
        raise to
    finally:
        if timeout is not None:
            signal.alarm(0)  # cancel alarm


import os


def _yes_or_no_or_always(question, default='y', timeout=None, always_option=True):
    """ Get Yes/No/Always answer with default choice and optional timeout

    :param question: prompt
    :param default: the default choice, i.e. 'y' or 'n' or 'a'
    :param timeout: the timeout in seconds, default is None. Does not work on windows.
    :param always_option: True to include, False to go to yes/no

    :returns: 'yes' or 'no' or 'always'
    """
    log = get_logger(_LOGGING_LEVEL)
    if default is not None and (default != 'y' and default != 'n'):
        raise ValueError(f'bad option for default: {default}; must be "y" or "n")')
    yes = 'Yes' if default == 'y' else 'yes'
    no = 'No' if default == 'n' else 'no'
    always = 'always'
    to_str = '' if timeout is None or os.name == 'nt' else f'(Timeout {default} in {timeout}s)'
    if always_option:
        quest_str = f'{question} {to_str} [{yes}/{no}/{always}]: '
    else:
        quest_str = f'{question} {to_str} [{yes}/{no}]: '

    while "the answer is invalid":
        try:
            if not timeout is None and os.name == 'nt':
                log.warning('cannot use timeout signal on windows')
                time.sleep(.1)  # make the warning come out first
                reply = str(input(quest_str)).lower().strip()
            else:
                reply = str(_input_with_timeout(quest_str, timeout=timeout)).lower().strip()
        except TimeoutError:
            log.warning(f'timeout expired, returning default={default} answer')
            reply = ''
        if len(reply) == 0 or reply == '':
            return yes.lower() if default == 'y' else no.lower()
        elif reply[0].lower() == 'y':
            return yes.lower()
        elif reply[0].lower() == 'n':
            return no.lower()
        elif always_option and reply[0].lower() == 'a':
            return always.lower()



dump_vars_func_string = """
import io
import dill
import logging
import fnmatch  # unix wildcard variable naming to save
from typing import Optional, Union  # variable typing for user hints

def __is_var(k, v):
    from types import ModuleType
    isvar = k.startswith('_') \
            or (isinstance(k, str) and (k == 'tmp' or k == 'In' or k == 'Out' or k == 'monkeypatch')) \
            or hasattr(v, '__call__') \
            or isinstance(v, ModuleType) \
            or isinstance(v, logging.Logger)
    return isvar

def __dump_vars_to_bytes(vars: Union[list, str, None] = None):
    import inspect, dill
    locals = None
    frame = inspect.currentframe().f_back
    try:
        locals = frame.f_locals
    finally:
        del frame
    if locals is None:
        return None

    if not vars is None:
        if not type(vars) is list and not type(vars) is str:
            raise ValueError(
                'vars argument must be a list of strings, e.g. ["a","b"], or a wildcard string, e.g. "a*", not vars'
            )

        vars_to_save = dict()  # a temp dict to save the ones we really want to save
        if type(vars) is list:
            def is_list_of_strings(lst):
                return bool(lst) and not isinstance(lst, str) and all(isinstance(elem, str) for elem in lst)

            if not is_list_of_strings(vars):
                raise ValueError('vars argument is a list but it must be a list of strings')

            # vars is a list of string names of vars
            for v in vars:
                if not v in locals.keys():
                    raise ValueError("v is not a local variable")
                else:
                    vars_to_save[v] = locals[v]
            locals = vars_to_save
        elif type(vars) is str:
            for v in locals:
                if fnmatch.fnmatch(v, vars):
                    vars_to_save[v] = locals[v]
        if len(vars_to_save) == 0:
            return None
        locals = vars_to_save

    data = {}
    could_not_pickle = []

    from types import ModuleType
    s = ''
    for k, v in locals.items():
        # don't try to pickle any objects that we don't want from notebook and anything that doesn't pickle
        if __is_var(k, v):
            continue
        try:
            if not dill.pickles(v):
                could_not_pickle.append(k)
                continue
        except:
            could_not_pickle.append(k)
            continue
        s = s + k + ' '
        data[k] = v
    s = s + ']'
    if len(data) == 0:
        return None

    buffer = io.BytesIO()
    try:
        try:
            dill.dump(data, buffer)
        except TypeError as e:
            pass
    except Exception as e:
        pass
    return buffer.getvalue()

import json
import base64

__json_out = {}
__internal_state = __dump_vars_to_bytes()
if not __internal_state:
    print(__json_out)
else:
    __encoded = base64.b64encode(__internal_state).decode()
    __json_out["base64"] = __encoded
    print(json.dumps(__json_out))
"""

dump_vars_func_string_v2 = """
import io
import json
import dill
import base64
import cloudpickle

def __is_serializable(obj):
    try:
        cloudpickle.dumps(obj)
        return True
    except Exception:
        return False

def __dump_vars_to_bytes():
    data = {k: v for k, v in globals().items() if not k.startswith('_') and __is_serializable(v)}
    buffer = io.BytesIO()
    try:
        try:
            dill.dump(data, buffer)
        except TypeError as e:
            pass
    except Exception as e:
        pass
    return buffer.getvalue()

__json_out = {}
__internal_state = __dump_vars_to_bytes()
if not __internal_state:
    print(__json_out)
else:
    __encoded = base64.b64encode(__internal_state).decode()
    __json_out["base64"] = __encoded
    print(json.dumps(__json_out))
"""

load_vars_func_string = """
import io
import dill
import base64
import logging
import fnmatch
from typing import Optional, Union

def load_vars_from_bytes(base64_string, overwrite: Union[str, bool] = 'yes'):
    bytes_data = base64.b64decode(base64_string)
    if not bytes_data:
        return None

    buffer = io.BytesIO(bytes_data)
    try:
        data = dill.load(buffer)
    except Exception as e:
        raise (e)

    import inspect
    try:
        frame = inspect.currentframe().f_back
        locals = frame.f_locals  # get its local variable dict
        always = False
        for k in data:
            try:
                if k in locals.keys():
                    if overwrite == 'yes' or always or ((type(overwrite) is bool) and overwrite):
                        locals[k] = data[k]
                        overwrote.append(k)
                    elif overwrite == 'prompt':
                        always = True
                        locals[k] = data[k]
                        overwrote.append(k)
                    else:
                        pass
                else:
                    locals[k] = data[k]
            except Exception as e:
                raise e
    finally:
        del frame

__internal_state_base64 = "{internal_state_base64}"
load_vars_from_bytes(__internal_state_base64)
"""

load_vars_func_string_v2 = """
import io
import dill
import base64

__internal_state_base64 = "{internal_state_base64}"
__bytes_data = base64.b64decode(__internal_state_base64)
__global_variables = dill.loads(__bytes_data)
globals().update(__global_variables)
"""