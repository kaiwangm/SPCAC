import sys
import os
import time

from model.core.rate_distortion import rate_distortion_loss
from model.zoo import load_model
import random
import statistics
from rich.progress import track
from utils.pcqm import pcqm
import tempfile
import uuid
import math

import torch
import argparse
import warnings
import numpy as np
from dataset import load_dataset

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


def write_ply_data(filename, points, colors):
    '''
    write data to ply file.
    '''
    if os.path.exists(filename):
        os.system('rm ' + filename)
    f = open(filename, 'a+')
    f.writelines(['ply\n', 'format ascii 1.0\n'])
    f.write('element vertex ' + str(points.shape[0]) + '\n')
    f.writelines([
        'property float x\n',
        'property float y\n',
        'property float z\n',
        'property uchar red\n',
        'property uchar green\n',
        'property uchar blue\n',
    ])
    f.write('end_header\n')
    for i, _ in enumerate(points):
        f.writelines([
            str(math.floor(points[i][0])), ' ',
            str(math.floor(points[i][1])), ' ',
            str(math.floor(points[i][2])), ' ',
            str(int(colors[i][0] * 255)), ' ',
            str(int(colors[i][1] * 255)), ' ',
            str(int(colors[i][2] * 255)), '\n',
        ])
    f.close()
    return


def main():
    # -------------------- argument -------------------
    parser = argparse.ArgumentParser()

    # Supported models: baseline_factorized baseline_mean grouping elpcac_l elpcac
    parser.add_argument('--model', type=str, default='baseline_factorized')
    parser.add_argument('--quality', type=int, default=3)
    parser.add_argument('--epoch', type=str, default='las')
    parser.add_argument('--train_dataset', type=str, default='coco3d')
    parser.add_argument('--dataset', type=str, default='j8ivfbv2-longdress10')
    parser.add_argument("--seed", type=int, default=777777,
                        help="Set random seed for reproducibility")
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument(
        '--skip_reconstruct', action='store_true', default=False,
        help='Skip writing reconstructed PLY files (faster for RD collection)',
    )

    opt = parser.parse_args()
    print(opt)

    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)

    evaluation(opt)


def get_model(opt):
    model, lam = load_model(
        opt.model, quality=opt.quality, category=opt.category)
    criterion = rate_distortion_loss(
        lam=lam, attributes=opt.attributes, metric=opt.metric)

    opt.method_name = model.__class__.__name__

    model_path = os.path.join('checkpoints', opt.train_dataset, opt.method_name, 'quality_{}'.format(
        opt.quality), 'eb_{}.pth'.format(opt.epoch))

    if not os.path.exists(model_path):
        # Try downloading from Hugging Face Hub as fallback
        try:
            from utils.hf_hub import download_checkpoint
            repo_filename = os.path.join(
                opt.train_dataset, opt.method_name,
                'quality_{}'.format(opt.quality),
                'eb_{}.pth'.format(opt.epoch)
            )
            model_path = download_checkpoint(repo_filename)
            print(f'Checkpoint downloaded from Hugging Face Hub: {model_path}')
        except ImportError:
            raise Exception(
                'model checkpoint file not found at {} '
                'and huggingface_hub is not installed'.format(model_path)
            )
        except RuntimeError as e:
            raise Exception(
                'model checkpoint file not found at {}. '
                'HF download also failed: {}'.format(model_path, e)
            )

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    model.cuda()
    model.update()
    model.eval()

    return model, criterion


def get_dataset(opt):
    dataset = load_dataset(opt.dataset, mode='test')
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    opt.dataset_name = opt.dataset

    return dataset, dataloader


def save_ply(points, colors, path):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd, write_ascii=True)


def get_parameter_number(net):

    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def _cuda_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _warmup_codec(model, points, colors, verbose=False):
    """Run one untimed compress/decompress to absorb CUDA init and kernel compile."""
    with torch.no_grad():
        out_enc = model.compress(points, colors)
        model.decompress(out_enc)
    _cuda_synchronize()
    if verbose:
        print('CUDA warmup done')


def evaluation(opt):
    torch.backends.cudnn.benchmark = True

    dataset, dataloader = get_dataset(opt)
    opt.category = dataset.category
    opt.attributes = dataset.attributes
    opt.metric = 'mse'

    if opt.model.startswith('mpeg'):
        return evaluation_mpeg(opt, dataset, dataloader)

    model, criterion = get_model(opt)

    print(get_parameter_number(model))

    out_log = {
        'bpp': [],
        'psnr_yuv': [],
        'psnr_y': [],
        'psnr': [],
        'ospcqm': [],
        'enc_time': [],
        'dec_time': [],
    }

    if hasattr(dataset, 'use_hdf5') and dataset.use_hdf5 == True:
        temp_dir = tempfile.mkdtemp()

    warmed_up = False

    # -------------------- train -------------------
    for batch_id, data in track(enumerate(dataloader), total=len(dataloader)):
        if hasattr(dataset, 'use_hdf5') and dataset.use_hdf5 == True:
            temp_name = str(uuid.uuid4())
            path = os.path.join(temp_dir, temp_name + '.ply')

            points, colors = data
            points = points.int()
            colors = colors.float()
            write_ply_data(
                path,
                points.squeeze(0).numpy(),
                colors.squeeze(0).numpy()
            )
        else:
            path = dataset.data_path_list[batch_id]
        
        if os.path.exists(path):
            print(path)

        # load point cloud
        points, colors = data
        points = points.int().cuda()
        colors = colors.float().cuda()

        with torch.no_grad():
            if not warmed_up:
                _warmup_codec(model, points, colors, verbose=opt.verbose)
                warmed_up = True

            _cuda_synchronize()
            enc_start = time.perf_counter()
            out_enc = model.compress(points, colors)
            _cuda_synchronize()
            out_enc['enc_time'] = time.perf_counter() - enc_start

            _cuda_synchronize()
            dec_start = time.perf_counter()
            out_dec = model.decompress(out_enc)
            _cuda_synchronize()
            out_dec['dec_time'] = time.perf_counter() - dec_start

            out_net = {**out_enc, **out_dec}

            out_criterion = criterion.evaluate(out_net)

        bpp = float(out_criterion['bpp'])
        if opt.attributes == 'RGB':
            psnr_yuv = float(out_criterion['psnr_yuv'])
            psnr_y = float(out_criterion['psnr_y'])
        elif opt.attributes == 'intensity':
            psnr = float(out_criterion['psnr'])

        enc_time = float(out_net['enc_time'])
        dec_time = float(out_net['dec_time'])

        if not getattr(opt, 'skip_reconstruct', False):
            points_hat = out_net['x_hat'].C[:, 1:]
            colors_hat = out_net['x_hat'].F
            colors_hat = torch.clip(colors_hat, 0.0, 1.0)

            ply_save_dir = os.path.join(
                './reconstructed', opt.dataset_name, opt.model, 'quality_{}'.format(opt.quality))
            if not os.path.exists(ply_save_dir):
                os.makedirs(ply_save_dir)

            if opt.attributes == 'RGB':
                save_ply(points_hat.detach().cpu().numpy(), colors_hat.detach().cpu(
                ).numpy(), os.path.join(ply_save_dir, '{}_q{}.ply'.format(batch_id, opt.quality)))

        out_log['bpp'].append(bpp)
        if opt.attributes == 'RGB':
            out_log['psnr_yuv'].append(psnr_yuv)
            out_log['psnr_y'].append(psnr_y)
            out_log['psnr'].append(0.0)
        elif opt.attributes == 'intensity':
            out_log['psnr_yuv'].append(0.0)
            out_log['psnr_y'].append(0.0)
            out_log['psnr'].append(psnr)

        out_log['enc_time'].append(enc_time)
        out_log['dec_time'].append(dec_time)

        org_pcqm = 1.0

        out_log['ospcqm'].append(1.0 - org_pcqm)

        if opt.verbose == True:
            if opt.attributes == 'RGB':
                print("batch_id: {}, bpp: {}, psnr_yuv: {}, psnr_y: {}, enc_time: {}, dec_time: {}".format(
                    batch_id, bpp, psnr_yuv, psnr_y, enc_time, dec_time)
                )
            elif opt.attributes == 'intensity':
                print("batch_id: {}, bpp: {}, psnr: {}, enc_time: {}, dec_time: {}".format(
                    batch_id, bpp, psnr, enc_time, dec_time)
                )

    if opt.attributes == 'RGB':
        print("mean bpp: {}, mean psnr_yuv: {}, mean psnr_y: {}".format(statistics.mean(
            out_log['bpp']), statistics.mean(out_log['psnr_yuv']), statistics.mean(out_log['psnr_y']))
        )
    elif opt.attributes == 'intensity':
        print("mean bpp: {}, mean psnr: {}".format(statistics.mean(
            out_log['bpp']), statistics.mean(out_log['psnr']))
        )
    print("mean enc_time: {}, mean dec_time: {}".format(statistics.mean(
        out_log['enc_time']), statistics.mean(out_log['dec_time']))
    )

    out_log_mean = {
        'bpp': statistics.mean(out_log['bpp']),
        'psnr_yuv': statistics.mean(out_log['psnr_yuv']),
        'psnr_y': statistics.mean(out_log['psnr_y']),
        'psnr': statistics.mean(out_log['psnr']),
        'ospcqm': statistics.mean(out_log['ospcqm']),
        'enc_time': statistics.mean(out_log['enc_time']),
        'dec_time': statistics.mean(out_log['dec_time']),
        'raw': out_log,
    }

    if hasattr(dataset, 'use_hdf5') and dataset.use_hdf5 == True:
        os.system('rm -rf ' + temp_dir)

    return out_log_mean


def evaluation_mpeg(opt, dataset, dataloader):
    import open3d as o3d
    from utils.gpcc_wrapper import gpcc_encode, gpcc_decode, get_psnr

    if hasattr(dataset, 'use_hdf5') and dataset.use_hdf5 == True:
        temp_dir = tempfile.mkdtemp()

    qp_lis = [51, 46, 40, 34, 28, 37]
    qp = qp_lis[opt.quality]

    base_dir = os.path.join('./reconstructed', opt.dataset_name,
                            opt.model, 'quality_{}'.format(opt.quality))

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    out_log = {
        'bpp': [],
        'psnr_yuv': [],
        'psnr_y': [],
        'psnr': [],
        'ospcqm': [],
        'enc_time': [],
        'dec_time': [],
    }

    for batch_id, data in track(enumerate(dataloader), total=len(dataloader)):
        temp_dir = tempfile.mkdtemp()
        temp_name = str(uuid.uuid4())
        path = os.path.join(temp_dir, temp_name + '.ply')

        points, colors = data
        points = points.int()
        colors = colors.float()
        write_ply_data(
            path,
            points.squeeze(0).numpy(),
            colors.squeeze(0).numpy()
        )

        if os.path.exists(path):
            print(path)

        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        ref_stream_size, enc_time = gpcc_encode(
            path, os.path.join(
                base_dir, 'tem_{}_q{}.bin'.format(batch_id, opt.quality)),
            qp=qp,
            show=opt.verbose,
            tmc13_version=opt.model
        )
        dec_time = gpcc_decode(
            os.path.join(base_dir, 'tem_{}_q{}.bin'.format(
                batch_id, opt.quality)),
            os.path.join(base_dir, '{}_q{}.ply'.format(batch_id, opt.quality)),
            show=opt.verbose,
            tmc13_version=opt.model
        )

        os.remove(os.path.join(
            base_dir, 'tem_{}_q{}.bin'.format(batch_id, opt.quality)))

        Num_points = points.shape[0]
        psnr_yuv, psnr_y = get_psnr(
            path,
            os.path.join(base_dir, '{}_q{}.ply'.format(batch_id, opt.quality)),
            show=opt.verbose
        )
        bpp = ref_stream_size * 8 / Num_points

        org_pcqm = 1.0

        if opt.attributes == 'RGB':
            org_pcqm = pcqm(path, os.path.join(
                base_dir, '{}_q{}.ply'.format(batch_id, opt.quality)), show=opt.verbose)

        out_log['bpp'].append(bpp)
        out_log['psnr_yuv'].append(float(psnr_yuv))
        out_log['psnr_y'].append(float(psnr_y))
        out_log['psnr'].append(0.0)
        out_log['ospcqm'].append(1.0 - org_pcqm)
        out_log['enc_time'].append(enc_time)
        out_log['dec_time'].append(dec_time)

    if opt.attributes == 'RGB':
        print("mean bpp: {}, mean psnr_yuv: {}, mean psnr_y: {}".format(statistics.mean(
            out_log['bpp']), statistics.mean(out_log['psnr_yuv']), statistics.mean(out_log['psnr_y']))
        )
    elif opt.attributes == 'intensity':
        print("mean bpp: {}, mean psnr: {}".format(statistics.mean(
            out_log['bpp']), statistics.mean(out_log['psnr']))
        )
    print("mean enc_time: {}, mean dec_time: {}".format(statistics.mean(
        out_log['enc_time']), statistics.mean(out_log['dec_time']))
    )

    out_log_mean = {
        'bpp': statistics.mean(out_log['bpp']),
        'psnr_yuv': statistics.mean(out_log['psnr_yuv']),
        'psnr_y': statistics.mean(out_log['psnr_y']),
        'psnr': statistics.mean(out_log['psnr']),
        'ospcqm': statistics.mean(out_log['ospcqm']),
        'enc_time': statistics.mean(out_log['enc_time']),
        'dec_time': statistics.mean(out_log['dec_time']),
        'raw': out_log,
    }

    if hasattr(dataset, 'use_hdf5') and dataset.use_hdf5 == True:
        os.system('rm -rf ' + temp_dir)

    return out_log_mean


if __name__ == '__main__':
    main()
