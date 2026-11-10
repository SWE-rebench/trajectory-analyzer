#!/usr/bin/env python3
"""Deterministic, payload-redacting command-method taxonomy v1."""
from __future__ import annotations
import re
import shlex
DEDICATED_EDIT={'edit','replace','multiedit','write','create','writefile'}
DEDICATED_READ={'read','open','view'}

def first_segment(command:str)->str:
 text=command.strip();line=text.splitlines()[0] if text else ''
 line=re.sub(r'^(?:cd\s+[^;&|]+\s*&&\s*)+','',line,flags=re.I)
 line=re.sub(r'^(?:env\s+)?(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)+','',line)
 line=re.sub(r'^(?:uv|poetry|pipenv)\s+run\s+','',line,flags=re.I)
 return line.strip().lower()
def flags(command:str)->dict[str,bool]:
 return {'heredoc':bool(re.search(r'<<\s*[\'\"]?[A-Za-z_][A-Za-z0-9_]*',command)),'output_redirection':bool(re.search(r'(?<![<>])>>?\s*(?![&])',command)),
  'python_shell':bool(re.match(r'^(?:/usr/bin/)?python(?:3(?:\.\d+)?)?(?:\s|$)',first_segment(command)))}
def explicit_pytest_quiet(command: str, runner: str) -> bool:
    """Return true only for an explicit pytest -q/-qq/--quiet option."""
    if runner not in {"pytest_direct", "python_m_pytest"}:
        return False
    try:
        lexer = shlex.shlex(first_segment(command), posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return False
    for token in tokens:
        if token in {"&&", "||", ";", "|", "&", "--"}:
            break
        if token == "--quiet" or re.fullmatch(r"-q+", token):
            return True
    return False

def edit_method(tool:str,command:str,action:str)->str:
 if action not in {'EDIT_SOURCE','CREATE_SOURCE','DELETE_SOURCE'}:return 'not_edit'
 t=tool.lower();h=first_segment(command)
 if t in DEDICATED_EDIT:return 'dedicated_edit_tool'
 if re.match(r'^(?:apply_patch|patch)(?:\s|$)',h) or t=='apply_patch':return 'apply_patch'
 if re.search(r'(?:^|\s)sed\s+(?:[^\n]*\s)?-i(?:\s|[.\'\"]|$)',h):return 'sed_inplace'
 if re.search(r'(?:^|\s)(?:perl|ruby)\s+[^\n]*(?:-[^\s]*i|\s-i)',h):return 'perl_or_ruby_inplace'
 if re.match(r'^(?:/usr/bin/)?python(?:3(?:\.\d+)?)?\s+(?:-c\b|-\b)',h) or (flags(command)['python_shell'] and flags(command)['heredoc']):return 'python_inline_mutation'
 if re.match(r'^(?:/usr/bin/)?python(?:3(?:\.\d+)?)?\s+\S+\.py(?:\s|$)',h):return 'python_script_mutation'
 if re.match(r'^(?:tee)(?:\s|$)',h):return 'tee_write'
 if re.match(r'^touch(?:\s|$)',h):return 'touch_create'
 if re.match(r'^(?:mv|cp)(?:\s|$)',h):return 'move_copy'
 if re.match(r'^(?:rm|unlink)(?:\s|$)',h):return 'delete_rm_unlink'
 if flags(command)['output_redirection']:return 'shell_redirection'
 return 'other_shell_mutation'
def read_method(tool:str,command:str,action:str)->str:
 if action not in {'READ_SOURCE','READ_TESTS'}:return 'not_read'
 t=tool.lower();h=first_segment(command)
 if t in DEDICATED_READ:return 'dedicated_read_tool'
 if re.match(r'^cat(?:\s|$)',h):return 'cat'
 if re.match(r'^sed\s+-n(?:\s|$)',h):return 'sed_print'
 if re.match(r'^(?:head|tail)(?:\s|$)',h):return 'head_tail'
 if re.match(r'^(?:nl|less)(?:\s|$)',h):return 'nl_less'
 if flags(command)['python_shell']:return 'python_read'
 return 'other_read'
def test_runner(command:str,action:str)->str:
 if action!='RUN_FORMAL_TEST':return 'not_test'
 h=first_segment(command)
 if re.match(r'^(?:/usr/bin/)?python(?:3(?:\.\d+)?)?\s+-m\s+pytest(?:\s|$)',h):return 'python_m_pytest'
 if re.match(r'^pytest(?:\s|$)',h):return 'pytest_direct'
 return 'project_or_custom_test'
def classify(tool:str,command:str,action:str)->dict[str,str]:
 return {'edit_method':edit_method(tool,command,action),
         'read_method':read_method(tool,command,action),
         'test_runner':test_runner(command,action)}
