#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:23:09 2019

@author: aparnami
"""

def get_args(arg_str):
    args = arg_str.split('\t\t')
    debug_args = []
    for arg in args[1:]:
        argsVal = arg.split()
        for item in argsVal:
            result = '"{}",'.format(item)
            debug_args.append(result)
    
    print('\n'.join(debug_args))
