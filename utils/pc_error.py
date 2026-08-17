import numpy as np
import os
import time
import pandas as pd
import subprocess
rootdir = os.path.split(__file__)[0]


def get_points_number(filedir):
    plyfile = open(filedir)

    line = plyfile.readline()
    while line.find("element vertex") == -1:
        line = plyfile.readline()
    number = int(line.split(' ')[-1][:-1])

    return number


def number_in_line(line):
    wordlist = line.split(' ')
    for _, item in enumerate(wordlist):
        try:
            number = float(item)
        except ValueError:
            continue

    return number


def pc_error(infile1, infile2, show=False):
    # Symmetric Metrics. D1 mse, D1 hausdorff.
    headers1 = [
        "mse1      (p2point)",
        "mse1,PSNR (p2point)",
        "c[0],    1         ",
        "c[1],    1         ",
        "c[2],    1         ",
    ]

    headers2 = [
        "mse2      (p2point)",
        "mse2,PSNR (p2point)",
        "c[0],    2         ",
        "c[1],    2         ",
        "c[2],    2         ",
    ]

    headersF = [
        "mseF      (p2point)",
        "mseF,PSNR (p2point)",
        "c[0],    F         ",
        "c[1],    F         ",
        "c[2],    F         ",
    ]

    headers = headers1 + headers2 + headersF

    command = str(
        './utils/bin/pc_error_d' +
        ' -a ' + infile1 +
        ' -b ' + infile2 +
        ' --color=1'
    )

    results = {}

    subp = subprocess.Popen(command,
                            shell=True, stdout=subprocess.PIPE)

    c = subp.stdout.readline()
    while c:
        line = c.decode(encoding='utf-8')
        if show:
            print(line)
        for _, key in enumerate(headers):
            if line.find(key) != -1:
                value = number_in_line(line)
                results[key] = value

        c = subp.stdout.readline()

    return pd.DataFrame([results])
