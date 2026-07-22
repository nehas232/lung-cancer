"""Load DICOM CT scans and convert pixel data to Hounsfield units."""

import numpy as np
import pydicom


def load_dicom_slice(filepath):
    """Read a single .dcm file and return (pixel_array_in_hu, dicom_dataset)."""
    ds = pydicom.dcmread(filepath)
    pixel_array = ds.pixel_array.astype(np.int16)

    intercept = getattr(ds, "RescaleIntercept", 0)
    slope = getattr(ds, "RescaleSlope", 1)
    hu_array = pixel_array * slope + intercept

    return hu_array.astype(np.int16), ds


def load_dicom_series(directory):
    """Load and z-sort all .dcm slices in a directory into a 3D volume."""
    import os

    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".dcm")]
    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    volume = np.stack([
        s.pixel_array.astype(np.int16) * getattr(s, "RescaleSlope", 1)
        + getattr(s, "RescaleIntercept", 0)
        for s in slices
    ])
    return volume.astype(np.int16)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        arr, ds = load_dicom_slice(sys.argv[1])
        print(f"Loaded slice with shape {arr.shape}, HU range [{arr.min()}, {arr.max()}]")