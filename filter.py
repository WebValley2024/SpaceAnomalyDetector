import xarray as xr
import os
from tqdm import tqdm
import normalizer as norm
import numpy as np


def process_directory(directory):
    corr = []
    filelist = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".zarr.zip"):
                filelist.append(os.path.join(root, f))
    for path in tqdm(filelist, desc="Processing files"):
        if int(path.split("/")[-1].split("_")[6][-1]) == 0:
            continue
        try:
            f = xr.open_zarr(path)
        except:
            continue
        add = True
        if path.split("/")[-2] == "EFD_ULF" and f.sizes["phony_dim_1"] != 42:
            continue
        try:
            if f["ALTITUDE"].isnull().any():
                continue
        except KeyError:
            if f["Altitude"].isnull().any():
                continue
        except:
            continue

        try:
            if f["UTC_TIME"].isnull().any():
                continue
        except KeyError:
            if f["UTCTime"].isnull().any():
                continue
        except:
            continue

        try:
            if f["GEO_LAT"].isnull().any():
                continue
            if f["GEO_LON"].isnull().any():
                continue
        except KeyError:
            if f["LonLat"].isnull().any():
                continue
        except:
            continue

        try:
            t = [norm.cses_to_unix(i) for i in f["UTC_TIME"].values.flatten()]
        except KeyError:
            t = [norm.cses_to_unix(i) for i in f["UTCTime"].values.flatten()]
        except:
            continue
        try:
            if (
                abs(t[-1] - t[0]) < 60
                or f.sizes["phony_dim_0"] < 60
                or any(abs(lat) > 90 for lat in f["GEO_LAT"].values)
                or any(abs(lon) > 180 for lon in f["GEO_LON"].values)
            ):
                continue

        except KeyError:
            if (
                abs(t[-1] - t[0]) < 60
                or f.sizes["phony_dim_0"] < 60
                or any(abs(lat) > 90 for lat in f["LonLat"].values[:, 1].squeeze())
                or any(abs(lon) > 180 for lon in f["LonLat"].values[:, 0].squeeze())
            ):
                continue

        for i in list(norm.mod_payload_params[norm.extract_payload(path)]):
            if i == "Count_Proton" and "Count_proton" in list(f.data_vars):
                i = "Count_proton"
            if i == "Count_Proton" and "HEPD_pro_counts" in list(f.data_vars):
                i = "HEPD_pro_counts"
            if i == "Count_Electron" and "Count_electron" in list(f.data_vars):
                i = "Count_electron"
            if i == "Count_Electron" and "HEPD_ele_counts" in list(f.data_vars):
                i = "HEPD_ele_counts"

            if f[i].isnull().any():
                add = False
                break

        if add:
            corr.append(path)

    return corr


def process_subdirectory(subdirectory, payload):
    dir = process_directory(subdirectory)
    np.save(f"working_files_{payload}.npy", dir)
    print(f"{payload}: {len(dir)}")

    return dir
