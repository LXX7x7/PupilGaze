#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export the EllSeg model to LibTorch and extract eye features from images."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from modelSummary import model_dict
from utils import get_predictions


class EllSegInferenceWrapper(torch.nn.Module):
    def __init__(self, model):
        super(EllSegInferenceWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        x4, x3, x2, x1, x = self.model.enc(x)
        latent = torch.mean(x.flatten(start_dim=2), -1)
        el_out = self.model.elReg(x, torch.tensor(0.0, dtype=x.dtype, device=x.device))
        seg_out = self.model.dec(x4, x3, x2, x1, x)
        return seg_out, latent, el_out


def load_model(model_name='ritnet_v3', weight_path='./weights/all.git_ok', device='cpu'):
    if model_name not in model_dict:
        raise ValueError(f'Unknown model name: {model_name}')

    model = model_dict[model_name]
    state = torch.load(weight_path, map_location='cpu')
    if 'state_dict' in state:
        model.load_state_dict(state['state_dict'], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    return model


def preprocess_frame(img, op_shape=(240, 320), align_width=True):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if align_width:
        if op_shape[1] != img.shape[1]:
            sc = op_shape[1] / img.shape[1]
            width = int(img.shape[1] * sc)
            height = int(img.shape[0] * sc)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)

            if op_shape[0] > img.shape[0]:
                pad_width = op_shape[0] - img.shape[0]
                if pad_width % 2 == 0:
                    img = np.pad(img, ((pad_width // 2, pad_width // 2), (0, 0)))
                else:
                    img = np.pad(img, ((np.floor(pad_width / 2), np.ceil(pad_width / 2)), (0, 0)))
                scale_shift = (sc, pad_width)

            elif op_shape[0] < img.shape[0]:
                pad_width = op_shape[0] - img.shape[0]
                if pad_width % 2 == 0:
                    img = img[-pad_width // 2:+pad_width // 2, ...]
                else:
                    img = img[-np.floor(pad_width / 2):+np.ceil(pad_width / 2), ...]
                scale_shift = (sc, pad_width)
            else:
                scale_shift = (sc, 0)
        else:
            scale_shift = (1, 0)
    else:
        raise ValueError('Height alignment is required for EllSeg preprocessing')

    img = (img.astype(np.float32) - img.mean()) / img.std()
    img = torch.from_numpy(img).unsqueeze(0).to(torch.float32)
    return img, scale_shift


def extract_eye_features(image_path, model, device='cpu', align_width=True):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f'Unable to open image file: {image_path}')

    input_tensor, _ = preprocess_frame(image, align_width=align_width)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        x4, x3, x2, x1, x = model.enc(input_tensor)
        latent = torch.mean(x.flatten(start_dim=2), -1)
        el_out = model.elReg(x, torch.tensor(0.0, dtype=x.dtype, device=x.device))
        seg_out = model.dec(x4, x3, x2, x1, x)

    return {
        'segmentation': get_predictions(seg_out).squeeze(0).cpu().numpy(),
        'latent_feature': latent.squeeze(0).cpu().numpy(),
        'ellipse_regression': el_out.squeeze(0).cpu().numpy()
    }


def export_libtorch_model(weight_path,
                           output_path,
                           model_name='ritnet_v3',
                           device='cpu'):
    model = load_model(model_name=model_name, weight_path=weight_path, device=device)
    wrapper = EllSegInferenceWrapper(model).to(device).eval()
    dummy_input = torch.randn(1, 1, 240, 320, dtype=torch.float32, device=device)
    traced = torch.jit.trace(wrapper, dummy_input, strict=False)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))
    return str(output_path)


def parse_args():
    parser = argparse.ArgumentParser(description='EllSeg LibTorch exporter and feature extractor')
    parser.add_argument('--command', type=str, choices=['export', 'extract'], default='export',
                        help='Select the operation to perform: export or extract')
    parser.add_argument('--model_name', type=str, default='ritnet_v3', help='EllSeg model name')
    parser.add_argument('--weights', type=str, default='./weights/all.git_ok', help='Path to model weights')
    parser.add_argument('--output', type=str, default='./weights/ellseg_ritnet_v3.pt', help='Path to save TorchScript model or extracted features')
    parser.add_argument('--image', type=str, default=None, help='Path to eye image for feature extraction')
    parser.add_argument('--export_device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device used for export or inference')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if args.export_device == 'cuda' and torch.cuda.is_available() else 'cpu')

    if args.command == 'export':
        output = export_libtorch_model(weight_path=args.weights,
                                       output_path=args.output,
                                       model_name=args.model_name,
                                       device=device)
        print(f'Saved LibTorch model to: {output}')

    elif args.command == 'extract':
        if args.image is None:
            raise ValueError('Please provide --image for extract command')
        model = load_model(model_name=args.model_name, weight_path=args.weights, device=device)
        features = extract_eye_features(args.image, model, device=device)
        np.save(args.output, features)
        print(f'Saved extracted features to: {args.output}')


if __name__ == '__main__':
    main()
