# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 14:36:19 2022

@author: asher
"""

import argparse
import glob
import os
import subprocess


OUTPUTDIR = r'E:\Shared\git\is-anime\train\anime'
OUTPUTCODE = r'%08d.bmp'


def build_command(finput, n_frames, foutput):
    command = list()
    command.extend([
        r'E:\Shared\Downloads\ffmpeg-n5.0-latest-win64-gpl-5.0\bin\ffmpeg.exe',
        r'-i',
        finput,
    ])

    if n_frames > 1:
        command.extend([
            r'-r',
            f'{n_frames}/1',
        ])

    command.extend([
        r'-vf',
        r'scale=128:128',
        foutput,
    ])

    return command


def main(pathtomovie='', n_frames=4):
    print('=' * 80)
    print(f'\tLooking in:\t\t\t{os.path.dirname(pathtomovie)}')
    print(f'\tSkipping frames:\t{n_frames}')
    print('=' * 80)

    fname = glob.glob(pathtomovie)

    for f in fname:
        outputsubdir = os.path.join(
            OUTPUTDIR, os.path.basename(os.path.splitext(f)[0]))
        if not os.path.exists(outputsubdir):
            os.mkdir(outputsubdir)

        command = build_command(
            f, n_frames, os.path.join(outputsubdir, '%08d.bmp'))

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
    parser.add_argument('n_frames', type=int, default=4, nargs='?',
                        help='dump a frame every <n_frames> frames')
    kwargs = vars(parser.parse_args())
    main(**kwargs)
