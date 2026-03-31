"""
Render 2 target views from a trained CoR-GS model.

Since CoR-GS uses ground-truth COLMAP poses (same coordinate frame as
metadata.json), no Sim(3) alignment is needed -- we just convert the
target c2w poses to the w2c format that 3DGS expects.

Usage:
    python render_targets.py \
        --source_path custom_Dataset_colmap \
        -m output/res_v12 \
        --cameras_json custom_Dataset/outputs/cameras.json \
        --iteration 40000
"""
import os
import json
import math
import torch
import torchvision
import numpy as np
from argparse import ArgumentParser

from scene import Scene
from scene.cameras import PseudoCamera
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--cameras_json", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    args = get_combined_args(parser)

    pipe = pipeline.extract(args)

    with open(args.cameras_json) as f:
        cam_data = json.load(f)

    target_c2ws = np.array(cam_data["camera_to_world"], dtype=np.float64)
    target_K = np.array(cam_data["camera_to_pixel"], dtype=np.float64)
    target_sizes = np.array(cam_data["image_size_xy"], dtype=np.float64)

    print(f"Loaded {len(target_c2ws)} target poses")

    with torch.no_grad():
        gaussians = GaussianModel(args)
        scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
        print(f"Point count: {gaussians.get_xyz.shape[0]}")

        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        out_dir = os.path.join(args.model_path, "target_renders")
        os.makedirs(out_dir, exist_ok=True)

        for i in range(len(target_c2ws)):
            c2w = target_c2ws[i]
            w2c = np.linalg.inv(c2w)

            R = w2c[:3, :3].T.astype(np.float32)
            T = w2c[:3, 3].astype(np.float32)

            W = int(target_sizes[i][0])
            H = int(target_sizes[i][1])
            fx = target_K[i][0][0]
            fy = target_K[i][1][1]
            fov_x = 2.0 * math.atan(W / (2.0 * fx))
            fov_y = 2.0 * math.atan(H / (2.0 * fy))

            view = PseudoCamera(
                R=R, T=T,
                FoVx=fov_x, FoVy=fov_y,
                width=W, height=H
            )

            rendering = render(view, gaussians, pipe, background)["render"]

            out_path = os.path.join(out_dir, f"output_{i}.png")
            torchvision.utils.save_image(rendering, out_path)
            print(f"Saved {out_path} ({W}x{H})")

    print(f"\nDone! Target renders saved to {out_dir}")


if __name__ == "__main__":
    main()
