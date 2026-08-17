import numpy as np
import os
import time
import pandas as pd
import subprocess
rootdir = os.path.split(__file__)[0]


def pcqm(infile1, infile2, show=False):
    command = str(
        './utils/bin/PCQM ' +
        '{} '.format(infile1) +
        '{} '.format(infile2) +
        '-r 0.004 ' +
        '-knn 20 ' +
        '-rx 2.0 ' +
        '--fastquit '
    )

    subp = subprocess.Popen(command,
                            shell=True, stdout=subprocess.PIPE)

    c = subp.stdout.readline()
    pcqm = 0.0
    while c:
        line = c.decode(encoding='utf-8')
        if show:
            print(line)

        stringc = str(c)
        if stringc.find('PCQM value is :') != -1:
            pcqm = float(stringc.split(' ')[-1][:-3])

        c = subp.stdout.readline()

    return pcqm
