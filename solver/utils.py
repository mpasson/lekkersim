#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 09:48:24 2021

@author: marco
"""

import types

def map_args(func, varsmap = {}):
    """ This function maps the argument names from a funciton to a new one
    
    Example: usnign this gunction on f(y, x=2) with mapping {'x' : 'a', 'y' : 'b'} will return f(a, b=2)
    Order of arguments and default values will be preserved
    
    Args:
        func (function):    Function which arguments need to be mapped
        varsmap (dict):     Dictionary of the mapping {'oldname' : 'newname'}
        
    Returns:
        function:   New function with the updated argument names
    """
    y=func
    oldargs = y.__code__.co_varnames[:y.__code__.co_argcount]
    new_args = []
    for x in oldargs:
        if x in varsmap:
            new_args.append(varsmap[x])
        else:
            new_args.append(x)
    new_args = tuple(new_args) + y.__code__.co_varnames[y.__code__.co_argcount:]
    print(new_args)
    y_code = types.CodeType(y.__code__.co_argcount, \
                y.__code__.co_kwonlyargcount, 
                y.__code__.co_nlocals, \
                y.__code__.co_stacksize, \
                y.__code__.co_flags, \
                y.__code__.co_code, \
                y.__code__.co_consts, \
                y.__code__.co_names, \
                #y.__code__.co_varnames, \
                new_args,
                y.__code__.co_filename, \
                y.__code__.co_name, \
                y.__code__.co_firstlineno, \
                y.__code__.co_lnotab)
    return types.FunctionType(y_code, y.__globals__, y.__name__, y.__defaults__)