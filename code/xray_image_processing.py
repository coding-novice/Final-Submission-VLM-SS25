import os
import numpy as np
import SimpleITK as sitk

REMOVE_BLACK_BARS = False # set to true if processing should also remove the black bars (Not recommended, messes with the aspect ratio)

# TODO adjust accordingly:
IN_DIR  = r"C:\Users\franc\OneDrive - TUM\1 - VLM - AI for Vision Lang Mod in Med Seminar\VLM_dataset\chest_xrays\images" 
if REMOVE_BLACK_BARS:
    # TODO adjust accordingly:
    OUT_DIR = r"C:\Users\franc\OneDrive - TUM\1 - VLM - AI for Vision Lang Mod in Med Seminar\VLM_dataset_processed\processed_chest_xray_images_no_black_bars"
else:
    # TODO adjust accordingly:
    OUT_DIR = r"C:\Users\franc\OneDrive - TUM\1 - VLM - AI for Vision Lang Mod in Med Seminar\VLM_dataset_processed\processed_chest_xray_images_with_black_bars"
TARGET            = 518
INTERP            = sitk.sitkBSpline
os.makedirs(OUT_DIR, exist_ok=True)

def find_anatomy_range(arr: np.ndarray):
    # Find leftmost and rightmost non-black column in the original (before cropping)
    mask = arr.max(axis=0) > 0
    if not mask.any():
        return 0, arr.shape[1] - 1  # edge case: image is all black
    left, right = np.where(mask)[0][[0, -1]]
    return left, right

def remove_side_bars(arr: np.ndarray):
    mask = arr.max(axis=0) > 0  
    if not mask.any():
        return arr
    left, right = np.where(mask)[0][[0, -1]]
    return arr[:, left:right+1]

def process_one(img: sitk.Image, orig_left, orig_right):
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim == 3:
        if arr.shape[-1] == 3: arr = arr[..., 0]
        elif arr.shape[0] == 3: arr = arr[0]
        else: raise ValueError(f"Bad shape {arr.shape}")
    arr = arr.astype(np.float32)

    # Save shape for pixel range calculation after cropping
    cropped_left, cropped_right = orig_left, orig_right

    # Remove black bars if requested
    if REMOVE_BLACK_BARS:
        mask = arr.max(axis=0) > 0
        if mask.any():
            cropped_left, cropped_right = np.where(mask)[0][[0, -1]]
        arr = remove_side_bars(arr)
    # Print info on pixels removed and remaining range in original coordinates:
    left_cut = cropped_left
    right_cut = arr.shape[1] - (cropped_right + 1)
    print(f"{fn}: removed {left_cut} px (left), {arr.shape[1] - (cropped_right - cropped_left + 1) - left_cut} px (right), "
          f"anatomy pixel range: x={cropped_left} to x={cropped_right}")

    tmp = sitk.GetImageFromArray(arr)
    tmp.SetSpacing(img.GetSpacing())
    W, H = tmp.GetSize()
    scale = TARGET / min(W, H)
    new_size = [int(W*scale), int(H*scale)]
    new_space = [sp/scale for sp in tmp.GetSpacing()]
    rf = sitk.ResampleImageFilter()
    rf.SetSize(new_size)
    rf.SetOutputSpacing(new_space)
    rf.SetInterpolator(INTERP)
    resized = rf.Execute(tmp)
    arr2 = sitk.GetArrayFromImage(resized).astype(np.float32)
    if not REMOVE_BLACK_BARS:
        mask = arr2 > 0
        if mask.any():
            vmin = float(arr2[mask].min())
            vmax = float(arr2.max())
        else:
            vmin, vmax = float(arr2.min()), float(arr2.max())
    else:
        vmin, vmax = float(arr2.min()), float(arr2.max())
    if vmax > vmin:
        arr2 = (arr2 - vmin)/(vmax - vmin)*255.0
    else:
        arr2.fill(0)
    arr2 = arr2.astype(np.uint8)
    H2, W2 = arr2.shape
    y0 = max((H2 - TARGET)//2, 0)
    x0 = max((W2 - TARGET)//2, 0)
    arr2 = arr2[y0:y0+TARGET, x0:x0+TARGET]
    return sitk.GetImageFromArray(arr2)

# Batch-process all PNGs
for fn in os.listdir(IN_DIR):
    if not fn.lower().endswith(".png"):
        continue
    inp = sitk.ReadImage(os.path.join(IN_DIR, fn))
    arr = sitk.GetArrayFromImage(inp)
    if arr.ndim == 3:
        if arr.shape[-1] == 3: arr = arr[..., 0]
        elif arr.shape[0] == 3: arr = arr[0]
    arr = arr.astype(np.float32)
    orig_left, orig_right = find_anatomy_range(arr)
    out = process_one(inp, orig_left, orig_right)
    sitk.WriteImage(out, os.path.join(OUT_DIR, fn))

print(f"Done—{'bars removed,' if REMOVE_BLACK_BARS else ''} all images are 518×518 PNGs.")
