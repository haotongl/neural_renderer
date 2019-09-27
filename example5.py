"""
Example 5. Finding object pose
"""
import os
import argparse
import glob
import cv2

import torch
import torch.nn as nn
import numpy as np
from skimage.io import imread, imsave
import tqdm
import imageio
import neural_renderer as nr
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

class Model(nn.Module):
    def __init__(self, filename_obj, filename_ref=None):
        super(Model, self).__init__()
        # load .obj
        #self.criterion = nn.CrossEntropyLoss(reduce=False)
        self.criterion = nn.MSELoss(reduce=False)
        vertices, faces = nr.load_obj(filename_obj)
        self.register_buffer('vertices', vertices[None, :, :])
        self.register_buffer('faces', faces[None, :, :])
        # create textures
        texture_size = 2
        textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.register_buffer('textures', textures)

        # load reference image
        image_ref = torch.from_numpy((imread(filename_ref)).astype(np.float32))
        self.register_buffer('image_ref', image_ref)

        # camera parameters
        self.K = torch.from_numpy(np.array([[280, 0., 128], [0., 280, 128], [0., 0., 1.]], dtype=np.float32)).cuda()
        T = np.load('pose0.npy')
        self.q = nn.Parameter(torch.from_numpy(np.load('q.npy')))
        #self.R = nn.Parameter(torch.from_numpy(T[:3, :3]))
        self.t = nn.Parameter(torch.from_numpy(T[:3, 3]))
        # setup renderer
        renderer = nr.Renderer(camera_mode='projection', image_size=256)
        renderer.K = self.K
        renderer.q = self.q
        renderer.t = self.t
        self.renderer = renderer

    def forward(self):
        image = self.renderer(self.vertices, self.faces, mode='silhouettes')
        loss = self.criterion(image[0, :, :], self.image_ref)
        loss = torch.mean(loss.view(loss.shape[0], -1), 1)
        return loss


def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imread(filename))
            os.remove(filename)
    writer.close()


def make_reference_image(filename_ref, filename_obj):
    model = Model(filename_obj)
    model.cuda()
    model.renderer.eye = nr.get_points_from_angles(2.732, 30, -15)
    images, _, _ = model.renderer.render(model.vertices, model.faces, torch.tanh(model.textures))
    image = images.detach().cpu().numpy()[0]
    imsave(filename_ref, image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, 'catply.obj'))
    parser.add_argument('-ir', '--filename_ref', type=str, default=os.path.join(data_dir, 'target.png'))
    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'example4_result.gif'))
    parser.add_argument('-mr', '--make_reference_image', type=int, default=0)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    if args.make_reference_image:
        make_reference_image(args.filename_ref, args.filename_obj)

    model = Model(args.filename_obj, args.filename_ref)
    model.cuda()

    # optimizer = chainer.optimizers.Adam(alpha=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loop = tqdm.tqdm(range(100))
    for i in loop:
        optimizer.zero_grad()
        loss = model()
        loss.mean().backward()
        optimizer.step()
        images = model.renderer(model.vertices, model.faces, mode='silhouettes')
        image = images.detach().cpu().numpy()[0]
        cv2.imwrite('/tmp/_tmp_%04d.png'%i, np.float32(image*255))
        loop.set_description('Optimizing (loss %.4f)' % loss.sum().item())
        if loss.sum().item() < 0.001:
            break
    make_gif(args.filename_output)


if __name__ == '__main__':
    main()
