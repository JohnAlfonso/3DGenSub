#!/usr/bin/env python3
import os
import sys
import argparse
import io
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from OpenGL.GL import GL_LINEAR
import trimesh
import pyrender


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# Constants (kept consistent with original script layout)
VIEWS_NUMBER = 16
THETA_ANGLES = np.linspace(0, 360, num=VIEWS_NUMBER)
PHI_ANGLES = np.full_like(THETA_ANGLES, -15.0)
GRID_VIEW_INDICES = [1, 5, 9, 13]  # 4 views for 2x2 grid
IMG_WIDTH = 518
IMG_HEIGHT = 518
GRID_VIEW_GAP = 5

# Camera settings
CAM_RAD = 2.5
CAM_FOV_DEG = 49.1


def spherical_to_cartesian(theta_deg: float, phi_deg: float, radius: float) -> np.ndarray:
    """Convert spherical coordinates (in degrees) to 3D Cartesian coordinates."""
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)

    x = radius * np.cos(phi) * np.cos(theta)
    y = radius * np.sin(phi)
    z = radius * np.cos(phi) * np.sin(theta)

    return np.array([x, y, z], dtype=np.float32)


def look_at(eye: np.ndarray, target: np.ndarray | None = None, up: np.ndarray | None = None) -> np.ndarray:
    """
    Build a camera pose matrix similar to utils.coords.look_at used in render_service/render.py.
    Returns a 4x4 world-space camera transform suitable for pyrender.
    """
    if target is None:
        target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    if up is None:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-8

    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-8

    up_vec = np.cross(right, forward)

    pose = np.eye(4, dtype=np.float32)
    pose[0, :3] = right
    pose[1, :3] = up_vec
    pose[2, :3] = -forward
    pose[:3, 3] = eye
    return pose


def combine_images4(images: list[Image.Image]) -> Image.Image:
    """Combine 4 PIL images into a 2x2 grid."""
    if len(images) != 4:
        raise ValueError(f"Expected 4 images to combine, got {len(images)}")

    row_width = IMG_WIDTH * 2 + GRID_VIEW_GAP
    column_height = IMG_HEIGHT * 2 + GRID_VIEW_GAP

    mode = images[0].mode
    bg_color = (0, 0, 0, 0) if "A" in mode else "black"

    combined_image = Image.new(mode, (row_width, column_height), color=bg_color)

    combined_image.paste(images[0], (0, 0))
    combined_image.paste(images[1], (IMG_WIDTH + GRID_VIEW_GAP, 0))
    combined_image.paste(images[2], (0, IMG_HEIGHT + GRID_VIEW_GAP))
    combined_image.paste(images[3], (IMG_WIDTH + GRID_VIEW_GAP, IMG_HEIGHT + GRID_VIEW_GAP))

    return combined_image


def render_glb_grid(glb_path: Path) -> Image.Image | None:
    """
    Render a GLB file into a 2x2 grid of views.

    This mirrors the logic of grid_from_glb_bytes in render-service/render.py:
    - load GLB with trimesh
    - convert to pyrender mesh, disable mipmaps
    - render multiple views with OffscreenRenderer (with SSAA + downsampling)
    - combine the 4 selected views into a 2x2 grid.
    """
    try:
        with open(glb_path, "rb") as f:
            glb_bytes = f.read()

        if not glb_bytes:
            print(f"[ERROR] Empty GLB file: {glb_path}")
            return None

        # Load mesh
        mesh = trimesh.load(
            file_obj=io.BytesIO(glb_bytes),
            file_type="glb",
            force="mesh",
        )

        # Create scene
        scene = pyrender.Scene(bg_color=[255, 255, 255, 0], ambient_light=[0.3, 0.3, 0.3])

        # Convert to pyrender mesh
        pyr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)

        # Disable mipmaps on all textures
        for primitive in pyr_mesh.primitives:
            if primitive.material is not None:
                mat = primitive.material
                for attr in [
                    "baseColorTexture",
                    "metallicRoughnessTexture",
                    "normalTexture",
                    "occlusionTexture",
                    "emissiveTexture",
                ]:
                    tex = getattr(mat, attr, None)
                    if tex is not None and hasattr(tex, "sampler") and tex.sampler is not None:
                        tex.sampler.minFilter = GL_LINEAR
                        tex.sampler.magFilter = GL_LINEAR

        scene.add(pyr_mesh)

        # Camera
        cam = pyrender.PerspectiveCamera(yfov=CAM_FOV_DEG * np.pi / 180.0)
        cam_node = scene.add(cam)

        # Light
        light = pyrender.DirectionalLight(color=[255, 255, 255], intensity=3.0)
        light_node = scene.add(light)

        # View angles (use same indexing strategy as original script)
        theta_angles = THETA_ANGLES[GRID_VIEW_INDICES].astype("float32")
        phi_angles = PHI_ANGLES[GRID_VIEW_INDICES].astype("float32")

        # 2x supersampling for antialiasing
        ssaa_factor = 2
        render_width = IMG_WIDTH * ssaa_factor
        render_height = IMG_HEIGHT * ssaa_factor
        renderer = pyrender.OffscreenRenderer(render_width, render_height)

        images: list[Image.Image] = []

        for theta, phi in zip(theta_angles, phi_angles):
            cam_pos = spherical_to_cartesian(theta, phi, CAM_RAD)
            pose = look_at(cam_pos)

            scene.set_pose(cam_node, pose)
            scene.set_pose(light_node, pose)

            image_array, _ = renderer.render(scene)

            # Downsample with high-quality Lanczos filter for antialiasing
            image_pil = Image.fromarray(image_array).resize(
                (IMG_WIDTH, IMG_HEIGHT),
                resample=Image.LANCZOS,
            )
            images.append(image_pil)

        if not images:
            print(f"[WARN] No views rendered for {glb_path}")
            return None

        grid = combine_images4(images)
        return grid

    except Exception as e:
        print(f"[ERROR] Failed to render GLB {glb_path}: {e}")
        return None


def process_directory(
    input_dir: Path,
    output_folder: Path,
    remaining: int | None,
) -> int:
    """Recursively render .glb files and mirror their structure inside the output root."""
    candidates = sorted(
        list(input_dir.rglob("*.glb")),
        key=lambda path: path.as_posix(),
    )

    if not candidates:
        print(f"[WARN] No .glb files found in {input_dir}")
        return 0

    processed = 0
    for glb_path in tqdm(candidates, desc=f"Rendering {input_dir.name}", unit="file"):
        if remaining is not None and remaining <= 0:
            break

        out_dir = output_folder / input_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{glb_path.stem}.png"

        if out_path.exists():
            processed += 1
            if remaining is not None:
                remaining -= 1
            continue

        try:
            grid = render_glb_grid(glb_path)
            if grid is not None:
                grid.save(out_path)
                processed += 1
                if remaining is not None:
                    remaining -= 1
            else:
                tqdm.write(f"[WARN] Failed to render {glb_path}")
        except Exception as e:
            tqdm.write(f"[WARN] Exception rendering {glb_path}: {e}")

    return processed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render 2x2 grids from GLB files while mirroring the source structure"
    )
    parser.add_argument(
        "--folders",
        type=str,
        nargs="+",
        default=None,
        help="One or more input folders to scan recursively (required)",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="./outputs/2x2_renders",
        help="Folder where rendered grids are stored",
    )
    parser.add_argument(
        "--N_instances",
        type=int,
        default=None,
        help="Maximum number of files to render (useful for debugging)",
    )
    args = parser.parse_args()

    assert args.folders is not None, "At least one folder must be provided"
    inputs = [Path(p) for p in args.folders]

    out_folder = Path(args.output_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    remaining = args.N_instances
    for in_folder in inputs:
        if remaining is not None and remaining <= 0:
            break
        if not in_folder.exists():
            sys.stderr.write(f"[WARN] Skipping missing directory: {in_folder}\n")
            continue
        processed = process_directory(
            in_folder,
            out_folder,
            remaining=remaining,
        )
        if remaining is not None:
            remaining -= processed


if __name__ == "__main__":
    main()