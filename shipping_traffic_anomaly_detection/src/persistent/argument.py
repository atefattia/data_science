import os
import argparse

def input_path_parser(add_help=False):
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument('input', action='store', type=os.path.abspath, help="the directory to read the scope from.")
    return parser

def output_path_parser(add_help=False):
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument('output', action='store', type=os.path.abspath, help="the directory to write the scope to.")
    return parser

def io_paths_parser(add_help=False):
    return argparse.ArgumentParser(parents=[input_path_parser(add_help), output_path_parser(add_help)],\
                                   add_help=add_help)
