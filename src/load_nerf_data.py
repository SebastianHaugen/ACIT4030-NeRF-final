import os
import json
import torch
import numpy as np
from PIL import Image
from pytorch3d.renderer import FoVPerspectiveCameras
import torch.nn.functional as F


def undistortion_grid(fx, fy, cx, cy, k1, k2, p1, p2, width, height, device):
    """Sampling grid for grid_sample that undistorts an image (OpenCV / COLMAP OPENCV model).

    The undistorted image gets square pixels with focal length fy in both directions and the
    principal point in the image centre, which is what FoVPerspectiveCameras assumes. For every
    pixel of that image we apply the lens distortion to find where it lies in the original photo,
    so no iterative inversion of the distortion is needed.
    """
    v, u = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32) + 0.5,
        torch.arange(width, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    # Normalised camera coordinates of the undistorted (ideal) image
    x = (u - width / 2) / fy
    y = (v - height / 2) / fy
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2
    x_d = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    y_d = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    # Pixel position in the original photo, then in grid_sample's [-1, 1] coordinates
    u_src = fx * x_d + cx
    v_src = fy * y_d + cy
    grid = torch.stack([u_src / width * 2 - 1, v_src / height * 2 - 1], dim=-1)
    return grid[None]  # (1, H, W, 2)


def load_nerf_data(data_dir, split="train", device="cpu", znear=0.01, zfar=10.0, resize_to=None,
                   premultiply_alpha=False, undistort=False):
    """
    Loads NeRF-style dataset (e.g., Lego, Nerfstudio output) from the specified directory and split.

    Args:
        data_dir (str): Path to the dataset directory
        split (str): Which split to load ('train', 'val', 'test').
        device (torch.device): The device to load tensors onto.
        znear (float): The near clip plane distance for the cameras.
        zfar (float): The far clip plane distance for the cameras.
        resize_to (tuple, optional): A tuple (height, width) to resize the images to.
                                     If None, images are loaded at their original resolution.
        premultiply_alpha (bool): Multiply RGBA colours by alpha (object over black, as rendered).
        undistort (bool): Remove lens distortion and use square pixels (needs fl_x, fl_y, k1, k2, p1, p2).

    Returns:
        tuple: A tuple containing:
            - target_images (torch.Tensor): Tensor of loaded images (N, H, W, 3).
            - target_silhouettes (torch.Tensor): Tensor of loaded silhouettes (N, H, W).
            - target_cameras (FoVPerspectiveCameras): PyTorch3D camera object.
    """
    # Load transforms.json dynamically
    transforms_filename_split = f"transforms_{split}.json"
    transforms_path_split = os.path.join(data_dir, transforms_filename_split)

    transforms_filename_default = "transforms.json"
    transforms_path_default = os.path.join(data_dir, transforms_filename_default)

    if os.path.exists(transforms_path_split):
        transforms_path = transforms_path_split
        print(f"Loading {transforms_filename_split}")
    elif os.path.exists(transforms_path_default):
        transforms_path = transforms_path_default
        print(f"Warning: {transforms_filename_split} not found. Loading {transforms_filename_default} instead.")
    else:
        raise FileNotFoundError(
            f"Neither {transforms_filename_split} nor {transforms_filename_default} found in {data_dir}."
        )

    with open(transforms_path, 'r') as f:
        transforms_data = json.load(f)

    all_images = []
    all_silhouettes = []
    Rs, Ts = [], []

    # Check for explicit intrinsic parameters (fl_x, fl_y, cx, cy, w, h)
    has_explicit_intrinsics = all(k in transforms_data for k in ['fl_x', 'fl_y', 'cx', 'cy', 'w', 'h'])
    has_camera_angle_x = 'camera_angle_x' in transforms_data

    # Always determine current image dimensions from the first actual image
    first_frame_file_path_relative = transforms_data['frames'][0]['file_path'].replace("./", "")
    first_image_path_base = os.path.join(data_dir, first_frame_file_path_relative)
    if not (first_image_path_base.endswith('.png') or first_image_path_base.endswith('.jpg') or first_image_path_base.endswith('.jpeg')):
        first_image_path = first_image_path_base + '.png'
    else:
        first_image_path = first_image_path_base

    first_image = Image.open(first_image_path)
    current_image_width, current_image_height = first_image.size

    # The 'w' and 'h' in transforms_data typically refer to the full resolution,
    # which we need to use for scaling intrinsics if explicit intrinsics are present.
    # If not present, we can just use the current_image_width/height as the base for scaling.
    full_res_w = transforms_data.get('w', current_image_width)
    full_res_h = transforms_data.get('h', current_image_height)

    # Final project fix: real cameras (poster) have lens distortion and slightly different focal
    # lengths in x and y, which FoVPerspectiveCameras cannot represent. The images are resampled
    # once at full resolution to an ideal pinhole camera with square pixels (focal length fl_y)
    undistort_grid = None
    distortion_keys = ('fl_x', 'fl_y', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2')
    if undistort and all(k in transforms_data for k in distortion_keys):
        s = current_image_width / full_res_w  # in case the images on disk are already downscaled
        undistort_grid = undistortion_grid(
            transforms_data['fl_x'] * s, transforms_data['fl_y'] * s,
            transforms_data['cx'] * s, transforms_data['cy'] * s,
            transforms_data['k1'], transforms_data['k2'], transforms_data['p1'], transforms_data['p2'],
            current_image_width, current_image_height, device,
        )
        print(f"Undistorting images (k1={transforms_data['k1']:.4f}, k2={transforms_data['k2']:.4f}, "
              f"p1={transforms_data['p1']:.4f}, p2={transforms_data['p2']:.4f}) to square pixels (f = fl_y)")

    # Determine target dimensions for resizing and FOV calculation
    if resize_to:
        target_height, target_width = resize_to
    else:
        target_height, target_width = current_image_height, current_image_width

    # Prepare camera intrinsics for PyTorch3D FoVPerspectiveCameras
    pytorch3d_fov = None # vertical FOV in degrees

    if has_explicit_intrinsics:
        # Scale intrinsics from transforms.json to match the target image dimensions
        scale_y = float(target_height) / float(full_res_h)
        fy_scaled = transforms_data['fl_y'] * scale_y

        # Calculate vertical FOV based on target dimensions and scaled focal length
        fov_y_radians = 2 * np.arctan(0.5 * target_height / fy_scaled)
        pytorch3d_fov = np.degrees(fov_y_radians)

        print(f"Using explicit intrinsics from transforms.json (scaled for {target_width}x{target_height}):")
        print(f"  fl_x={transforms_data['fl_x'] * (float(target_width)/float(full_res_w)):.2f}, fl_y={fy_scaled:.2f}, cx={transforms_data['cx'] * (float(target_width)/float(full_res_w)):.2f}, cy={transforms_data['cy'] * scale_y:.2f}")

    elif has_camera_angle_x:
        # Fallback to camera_angle_x
        camera_angle_x = transforms_data['camera_angle_x']
        # Focal length calculation based on original image width, then scaled for target height
        focal_length_original_x = 0.5 * full_res_w / np.tan(0.5 * camera_angle_x)
        focal_length_target_y = focal_length_original_x * (float(target_height) / float(full_res_h))
        fov_y_radians = 2 * np.arctan(0.5 * target_height / focal_length_target_y)
        pytorch3d_fov = np.degrees(fov_y_radians)
        print(f"Using camera_angle_x: {camera_angle_x:.3f} rad, calculated FOV: {pytorch3d_fov:.3f} deg")

    else:
        raise ValueError("Neither explicit intrinsics (fl_x, fl_y, etc.) nor camera_angle_x found in transforms.json.")

    print(f"Loading {split} split with {len(transforms_data['frames'])} frames.")
    print(f"Loaded image dimensions: {current_image_width}x{current_image_height}")
    if resize_to:
        print(f"Resizing images to: {target_width}x{target_height}")
    elif (current_image_width != full_res_w) or (current_image_height != full_res_h):
        print(f"Original transforms.json dimensions: {full_res_w}x{full_res_h}. Loaded images are already {current_image_width}x{current_image_height}.")

    print(f"Calculated Vertical FOV for camera: {pytorch3d_fov:.3f} degrees")

    for frame in transforms_data['frames']:
        # Extract file_path from JSON, remove './' if present
        file_path_from_json = frame['file_path'].replace("./", "")

        # Construct the image path by joining data_dir with the relative file_path from JSON
        # and ensure it has a .png extension if not already present.
        img_path_base = os.path.join(data_dir, file_path_from_json)
        if not (img_path_base.lower().endswith('.png') or img_path_base.lower().endswith('.jpg') or img_path_base.lower().endswith('.jpeg')):
            img_path = img_path_base + '.png'
        else:
            img_path = img_path_base

        # Load image
        img = Image.open(img_path)
        img_np = np.array(img).astype(np.float32) / 255.0

        # Determine if the image has an alpha channel and extract RGB/silhouette accordingly
        if img_np.shape[-1] == 4:
            # RGBA image
            image_rgb = img_np[..., :3]
            silhouette = img_np[..., 3]  # Alpha channel as silhouette
        elif img_np.shape[-1] == 3:
            # RGB image, create a fully opaque silhouette
            image_rgb = img_np
            silhouette = np.ones(img_np.shape[:-1], dtype=np.float32) # Fully opaque silhouette
        else:
            raise ValueError(f"Unsupported image format: {img_np.shape[-1]} channels found in {img_path}")

        # Convert to tensor and apply resizing if specified
        image_rgb_tensor = torch.from_numpy(image_rgb).to(device) # (H, W, C)
        silhouette_tensor = torch.from_numpy(silhouette).to(device) # (H, W)

        # Final project fix: PNG colours are not multiplied by alpha, but the renderer composites
        # onto black, so semi-transparent edge pixels could never be matched. Premultiplying
        # gives the target the model can actually render (the object over a black background)
        if premultiply_alpha:
            image_rgb_tensor = image_rgb_tensor * silhouette_tensor[..., None]

        if undistort_grid is not None:
            image_rgb_tensor = F.grid_sample(
                image_rgb_tensor.permute(2, 0, 1)[None], undistort_grid,
                mode='bilinear', padding_mode='border', align_corners=False,
            )[0].permute(1, 2, 0)

        if resize_to:
            # Reshape from (H, W, C) to (1, C, H, W) for interpolation
            image_rgb_tensor = image_rgb_tensor.permute(2, 0, 1).unsqueeze(0) # (1, C, H, W)
            # Reshape from (H, W) to (1, 1, H, W) for interpolation
            silhouette_tensor = silhouette_tensor.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)

            # Resize using bicubic interpolation
            image_rgb_tensor = F.interpolate(
                image_rgb_tensor, size=resize_to, mode='bicubic', align_corners=False
            ).squeeze(0).permute(1, 2, 0) # Back to (H', W', C)

            silhouette_tensor = F.interpolate(
                silhouette_tensor, size=resize_to, mode='bicubic', align_corners=False
            ).squeeze(0).squeeze(0) # Back to (H', W')

        all_images.append(image_rgb_tensor.cpu()) 
        all_silhouettes.append(silhouette_tensor.cpu()) 

        # Extract transform matrix (camera-to-world)
        c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32, device=device)

        # Convert camera-to-world to PyTorch3D R, T (world-to-camera)
        R_nerf = c2w[:3, :3] # Rotation part of camera-to-world
        T_nerf = c2w[:3, 3] # Translation part of camera-to-world

        flip = torch.diag(torch.tensor([-1.0, 1.0, -1.0], device=device))  # OpenGL -> PyTorch3D axes
        R_pytorch3d = R_nerf @ flip
        T_pytorch3d = -R_pytorch3d.T @ T_nerf

        Rs.append(R_pytorch3d)
        Ts.append(T_pytorch3d)

    target_images = torch.stack(all_images, dim=0).to(device)
    target_silhouettes = torch.stack(all_silhouettes, dim=0).to(device)
    target_Rs = torch.stack(Rs, dim=0).to(device)
    target_Ts = torch.stack(Ts, dim=0).to(device)

    # Create PyTorch3D FoVPerspectiveCameras object using fov
    target_cameras = FoVPerspectiveCameras(
        R=target_Rs,
        T=target_Ts,
        fov=pytorch3d_fov, 
        degrees=True,
        znear=znear,
        zfar=zfar,
        device=device,
    )

    return target_images, target_silhouettes, target_cameras