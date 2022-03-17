# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 14:36:19 2022

@author: asher
"""

import argparse
import glob
import os
import subprocess


OUTPUTDIR = './'
OUTPUTCODE = '%05d.jpg'


def build_command(finput, foutput):
    command = list()
    command.extend([
        r'ffmpeg.exe',  # symlink to (static!) ffmpeg in local dir
        r'-hwaccel_output_format',
        r'cuda',
        r'-i',
        finput,
        r'-r',
        r'1/1',  # force 1 fps input to 1 fps output
        r'-vf',
        # need larger frame size for feature detection (maybe)
        r'scale=512:512',
        foutput,
    ])

    return command


def main(pathtomovie='', mediatype=False):
    print('=' * 80)
    print(f'\tLooking in:\t\t\t{os.path.dirname(pathtomovie)}')
    print('=' * 80)

    fname = glob.glob(pathtomovie, recursive=True)

    for f in fname:
        outputsubdir = os.path.join(
            OUTPUTDIR, 'anime' if mediatype is True else 'photo',
            os.path.basename(os.path.splitext(f)[0]))
        if not os.path.exists(outputsubdir):
            os.mkdir(outputsubdir)

        command = build_command(
            f, os.path.join(outputsubdir, OUTPUTCODE))

        with subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True, shell=True) as p:
            for line in p.stdout:
                print(line, end='')

        print('\n' + '=' * 80, end='\n' * 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract frames from a video file.'
    )
    parser.add_argument('pathtomovie', type=str,
                        help='glob string for movie name')
    parser.add_argument('-a', '--anime', dest='mediatype', action='store_true',
                        default=False, help='input media type')
    kwargs = vars(parser.parse_args())
    main(**kwargs)
