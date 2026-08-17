import os
import numpy as np
import subprocess
import math
from utils.pc_error import pc_error


def gpcc_encode(filedir, bin_dir, show=False, qp=8, tmc13_version='mpeg-tmc13v19'):
    """Compress point cloud losslessly using MPEG G-PCC. 
    You can download and install TMC13 from 
    http://mpegx.int-evry.fr/software/MPEG/PCC/TM/mpeg-pcc-tmc13
    """

    if tmc13_version == 'mpeg-tmc13v23':
        subp = subprocess.Popen('./utils/bin/{}'.format(tmc13_version) +
                                ' --mode=0' +
                                ' --trisoupNodeSizeLog2=0' +
                                ' --mergeDuplicatedPoints=0' +
                                ' --neighbourAvailBoundaryLog2=8' +
                                ' --intra_pred_max_node_size_log2=6' +
                                ' --positionQuantizationScale=1' +

                                ' --inferredDirectCodingMode=1' +

                                ' --maxNumQtBtBeforeOt=4' +
                                ' --minQtbtSizeLog2=0' +
                                ' --planarEnabled=1' +
                                ' --planarModeIdcmUse=0' +

                                ' --convertPlyColourspace=1' +
                                ' --transformType=0' +

                                ' --qp={}'.format(qp) +
                                ' --qpChromaOffset=0' +
                                ' --bitdepth=8' +
                                ' --attrOffset=0' +
                                ' --attrScale=1' +
                                ' --attribute=color' +

                                ' --uncompressedDataPath=' + filedir +
                                ' --compressedStreamPath=' + bin_dir,
                                shell=True, stdout=subprocess.PIPE
                                )
    elif tmc13_version == 'mpeg-tmc13v19':
        subp = subprocess.Popen('./utils/bin/{}'.format(tmc13_version) +
                                ' --mode=0' +
                                ' --trisoupNodeSizeLog2=0' +
                                ' --mergeDuplicatedPoints=0' +
                                ' --neighbourAvailBoundaryLog2=8' +
                                ' --intra_pred_max_node_size_log2=6' +
                                ' --positionQuantizationScale=1' +

                                ' --inferredDirectCodingMode=1' +

                                ' --maxNumQtBtBeforeOt=4' +
                                ' --minQtbtSizeLog2=0' +
                                ' --planarEnabled=1' +
                                ' --planarModeIdcmUse=0' +

                                ' --convertPlyColourspace=1' +
                                ' --transformType=0' +

                                ' --qp={}'.format(qp) +
                                ' --qpChromaOffset=0' +
                                ' --bitdepth=8' +
                                ' --attrOffset=0' +
                                ' --attrScale=1' +
                                ' --attribute=color' +

                                ' --uncompressedDataPath=' + filedir +
                                ' --compressedStreamPath=' + bin_dir,
                                shell=True, stdout=subprocess.PIPE
                                )
    elif tmc13_version == 'mpeg-tmc13v6':
        subp = subprocess.Popen('./utils/bin/{}'.format(tmc13_version) +
                                ' --mode=0' +
                                ' --trisoup_node_size_log2=0' +
                                ' --mergeDuplicatedPoints=0' +
                                ' --ctxOccupancyReductionFactor=3' +
                                ' --neighbourAvailBoundaryLog2=8' +
                                ' --intra_pred_max_node_size_log2=6' +
                                ' --positionQuantizationScale=1' +

                                ' --colorTransform=1' +
                                ' --transformType=1' +

                                ' --rahtLeafDecimationDepth=0' +
                                ' --qp={}'.format(qp) +
                                ' --qpChromaOffset=0' +
                                ' --bitdepth=8' +

                                ' --attribute=color' +

                                ' --uncompressedDataPath=' + filedir +
                                ' --compressedStreamPath=' + bin_dir,
                                shell=True, stdout=subprocess.PIPE
                                )

    c = subp.stdout.readline()
    ref_stream_size = 0.0
    enc_time = 0.0
    while c:
        if show:
            print(c)

        stringc = str(c)
        if stringc.find('colors bitstream size') != -1:
            ref_stream_size += float(stringc.split(' ')[3])
        if stringc.find('colors processing time') != -1:
            enc_time += float(stringc.split(' ')[4])
        c = subp.stdout.readline()

    return ref_stream_size, enc_time


def gpcc_decode(bin_dir, rec_dir, show=False, tmc13_version='tmc13v19'):

    if tmc13_version == 'mpeg-tmc13v23':
        subp = subprocess.Popen('./utils/bin/{}'.format(tmc13_version) +
                                ' --mode=1' +
                                ' --outputBinaryPly=0' +
                                ' --convertPlyColourspace=1' +
                                ' --compressedStreamPath=' + bin_dir +
                                ' --reconstructedDataPath=' + rec_dir,
                                shell=True, stdout=subprocess.PIPE)
    if tmc13_version == 'mpeg-tmc13v19':
        subp = subprocess.Popen('./utils/bin/{}'.format(tmc13_version) +
                                ' --mode=1' +
                                ' --outputBinaryPly=0' +
                                ' --convertPlyColourspace=1' +
                                ' --compressedStreamPath=' + bin_dir +
                                ' --reconstructedDataPath=' + rec_dir,
                                shell=True, stdout=subprocess.PIPE)
    elif tmc13_version == 'mpeg-tmc13v6':
        subp = subprocess.Popen('./utils/bin/{}'.format(tmc13_version) +
                                ' --mode=1' +
                                ' --outputBinaryPly=0' +
                                ' --colorTransform=1' +
                                ' --compressedStreamPath=' + bin_dir +
                                ' --reconstructedDataPath=' + rec_dir,
                                shell=True, stdout=subprocess.PIPE)

    c = subp.stdout.readline()
    dec_time = 0.0
    while c:
        if show:
            print(c)
        c = subp.stdout.readline()

        stringc = str(c)
        if stringc.find('colors processing time') != -1:
            dec_time += float(stringc.split(' ')[4])

    return dec_time


def avs_pcc_encode(filedir, bin_dir, show=False):
    """Compress point cloud losslessly using MPEG G-PCCv6. 
    You can download and install TMC13 from 
    http://mpegx.int-evry.fr/software/MPEG/PCC/TM/mpeg-pcc-tmc13
    """

    subp = subprocess.Popen('./utils/bin/avs-pcc-encoder ' +
                            ' -i ' + filedir +
                            ' -b ' + bin_dir +
                            ' -gqs=1 ' +
                            ' -gof=1',
                            shell=True, stdout=subprocess.PIPE)
    c = subp.stdout.readline()
    while c:
        if show:
            print(c)
        c = subp.stdout.readline()

    return


def avs_pcc_decode(bin_dir, rec_dir, show=False):
    subp = subprocess.Popen('./utils/bin/avs-pcc-decoder ' +
                            ' -b ' + bin_dir +
                            ' -r ' + rec_dir,
                            shell=True, stdout=subprocess.PIPE)
    c = subp.stdout.readline()
    while c:
        if show:
            print(c)
        c = subp.stdout.readline()

    return


def load_ply_data(filename):
    '''
    load data from ply file.
    '''

    f = open(filename)
    # 1.read all points
    points = []
    colors = []
    for line in f:
        # only x,y,z
        wordslist = line.split(' ')
        try:
            x, y, z = float(wordslist[0]), float(
                wordslist[1]), float(wordslist[2])
            r, g, b = float(wordslist[3]), float(
                wordslist[4]), float(wordslist[5])
        except ValueError:
            continue
        points.append([x, y, z])
        colors.append([r, g, b])
    points = np.array(points)
    colors = np.array(colors)
    points = points.astype(np.int32)
    colors = colors.astype(np.uint8)
    f.close()

    return points, colors


def write_ply_data(filename, points):
    '''
    write data to ply file.
    '''
    if os.path.exists(filename):
        os.system('rm ' + filename)
    f = open(filename, 'a+')
    f.writelines(['ply\n', 'format ascii 1.0\n'])
    f.write('element vertex ' + str(points.shape[0]) + '\n')
    f.writelines(['property float x\n',
                  'property float y\n',
                  'property float z\n',
                  'property uchar red\n',
                  'property uchar green\n',
                  'property uchar blue\n',
                  'property uchar alpha\n'
                  ])
    f.write('end_header\n')
    for _, point in enumerate(points):
        f.writelines([
            str(math.floor(point[0])), ' ',
            str(math.floor(point[1])), ' ',
            str(math.floor(point[2])), ' ',
            str(int(point[3] * 255)), ' ',
            str(int(point[4] * 255)), ' ',
            str(int(point[5] * 255)), ' ',
            str(int(255.0)),
            '\n'])
    f.close()

    return


def get_psnr(filename_a, filename_b, show=False):

    psnr_out = pc_error(filename_a, filename_b, show=show)
    mse_yuv = (4.0 * psnr_out['c[0],    F         '] + psnr_out['c[1],    F         '] + psnr_out['c[2],    F         ']) / 6.0
    mse_y = psnr_out['c[0],    F         ']

    psnr_yuv = 10.0 * np.log10((1.0 * 1.0) / mse_yuv)
    psnr_y = 10.0 * np.log10((1.0 * 1.0) / mse_y)

    return psnr_yuv, psnr_y
