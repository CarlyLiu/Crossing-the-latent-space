"""Generate images using pretrained network pickle."""
"""python gen_lerp_images.py --outdir=try4 --seed-a=432 --seed-b=143 --num-steps=5"""


import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

@click.command()
# @click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
# @click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
# @click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0', show_default=True, metavar='VEC2')
# @click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, show_default=True, metavar='ANGLE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--seed-a', help='First seed for interpolation', type=int, required=True, metavar='SEED_A')
@click.option('--seed-b', help='Second seed for interpolation', type=int, required=True, metavar='SEED_B')
@click.option('--num-steps', help='Number of interpolation steps', type=int, required=True, metavar='NUM_STEPS')

def generate_images(
    # network_pkl: str,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    seed_a: int,
    seed_b: int,
    num_steps: int
):
    network_pkl = r'C:\Users\project\stylegan3\stylegan3-r-ffhqu-256x256.pkl'
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    def lerp(u, v, a):
        # 线性插值
        return a * u + (1 - a) * v

    # 根据种子生成随机潜在向量
    np.random.seed(seed_a)
    latent_a = np.random.randn(1, G.z_dim)  # G.z_dim 是潜在向量的维度

    np.random.seed(seed_b)
    latent_b = np.random.randn(1, G.z_dim)

    os.makedirs(outdir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    # 生成并保存插值图像
    for i in range(num_steps + 1):
        # 计算插值系数
        interpolation_ratio = i / num_steps

        # 进行线性插值
        interpolated_latent = lerp(latent_a, latent_b, interpolation_ratio)

        # 将插值后的潜在向量转换为图像
        with torch.no_grad():
            z = torch.from_numpy(interpolated_latent).to(device)
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        # 处理并保存图像
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/interpolated_image_{i:03d}.png')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
