import datetime
import xarray as xr
import numpy as np
import os
from tqdm import tqdm
from numcodecs import Blosc

mod_payload_params = {
    "HEP_1": ["Count_Electron", "Count_Proton", "A411", "A412"],
    "HEP_2": ["Count_Electron", "Count_Proton", "A411", "A412"],
    "HEP_3": ["Count_Proton", "Count_Electron"],
    "HEP_4": ["XrayRate"],
    "EFD_1": ["A111_W", "A112_W", "A113_W", "A111_P", "A112_P", "A113_P"],
    "LAP_1": ["A311", "A321"],
}


def extract_payload(filename: str):
    """
    Extract payload name from filename.
    :param filename: The filename to extract the payload name from.
    :return: The payload name
    """
    filename = filename.split("/")[-1]
    if filename.startswith("CSES_HEP_DDD"):
        return "HEP_3"
    return filename.split("_")[2] + "_" + filename.split("_")[3]


def reformat_specific(
    payload: str, var_name: str, var_data: xr.DataArray, new_vars: dict
):
    """
    Reformats a specific feature in a payload Dataset.
    :param payload: The payload's name
    :param var_name: The name of the feature
    :param var_name: The feature's data
    :param new_vars: A dictionary containing the new data
    :return: The new DataArray
    """

    if payload not in mod_payload_params.keys():
        raise ValueError(f"{payload} not found in known payloads.")

    if var_name not in mod_payload_params[payload] + [
        "GEO_LAT",
        "GEO_LON",
        "UTC_TIME",
        "ALTITUDE",
    ]:
        raise ValueError(
            f"{var_name} not found in known features for payload {payload}."
        )

    if var_name in ["GEO_LAT", "GEO_LON", "UTC_TIME", "ALTITUDE"]:
        new_vars[var_name] = var_data[:].squeeze()

    elif payload in ["HEP_1", "HEP_2"]:
        if var_name in ["Count_Electron", "Count_Proton"]:
            if payload == "HEP_1":
                for i in range(var_data.sizes["phony_dim_2"]):
                    new_vars[var_name + "_" + str(i)] = var_data[:, i]
            elif payload == "HEP_2":
                new_vars[var_name] = var_data[:].squeeze()
        elif var_name in ["A411", "A412"]:
            if payload == "HEP_1":
                for j in range(var_data.sizes["phony_dim_2"]):
                    for index, i in enumerate(
                        range(0, var_data.sizes["phony_dim_1"] - 1, 85)
                    ):
                        new_vars[var_name + "_" + str(index) + "_" + str(j)] = var_data[
                            :, i : i + 85, j
                        ].mean(dim="phony_dim_1")
            elif payload == "HEP_2":
                for jindex, j in enumerate(range(0, var_data.sizes["phony_dim_2"], 4)):
                    for index, i in enumerate(
                        range(0, var_data.sizes["phony_dim_1"] - 1, 85)
                    ):
                        new_vars[var_name + "_" + str(index) + "_" + str(jindex)] = (
                            var_data[:, i : i + 85, j : j + 4]
                            .mean(dim="phony_dim_2")
                            .mean(dim="phony_dim_1")
                        )
    elif payload == "HEP_3":
        if var_name in ["Count_Electron", "Count_Proton"]:
            new_vars[var_name] = var_data[:].squeeze()

    elif payload == "HEP_4":
        new_vars[var_name] = var_data[:].squeeze()

    elif payload == "EFD_1":
        if var_name in ["A111_W", "A112_W", "A113_W"]:
            new_vars[var_name] = var_data.mean(dim="phony_dim_2")

        elif var_name in ["A111_P", "A112_P", "A113_P"]:
            for index, i in enumerate(range(0, var_data.sizes["phony_dim_1"] - 2, 4)):
                new_vars[var_name + "_" + str(index)] = var_data[:, i : i + 12].mean(
                    dim="phony_dim_1"
                )

    elif payload == "LAP_1":
        new_vars[var_name] = var_data[:].squeeze()

    return new_vars


def reformat_xr(fil: xr.Dataset):
    """
    Reformats dataset according to rules.
    :param fil: The Dataset
    :return: The reformatted xarray Dataset
    """

    new_vars = {}

    name = extract_payload(fil.encoding["source"])

    if name not in mod_payload_params.keys():
        raise ValueError(f"{name} not found in known payloads.")

    if name == "HEP_3":
        fil = fil.rename(
            {
                "Altitude": "ALTITUDE",
                "HEPD_ele_counts": "Count_Electron",
                "HEPD_pro_counts": "Count_Proton",
                "UTCTime": "UTC_TIME",
            }
        )
        fil["GEO_LAT"] = xr.DataArray(
            fil["LonLat"].values[:, 1].squeeze(), dims="phony_dim_0"
        )
        fil["GEO_LON"] = xr.DataArray(
            fil["LonLat"].values[:, 0].squeeze(), dims="phony_dim_0"
        )

    for var_name, var_data in fil.data_vars.items():
        if var_name == "Count_proton":
            fil = fil.rename({"Count_proton": "Count_Proton"})
        if var_name == "Count_electron":
            fil = fil.rename({"Count_electron": "Count_Electron"})

        if var_name not in mod_payload_params[name] + [
            "GEO_LAT",
            "GEO_LON",
            "UTC_TIME",
            "ALTITUDE",
        ]:
            continue

        dims = var_data.dims

        has_phony_dims_0 = "phony_dim_0" in dims

        if has_phony_dims_0:
            if "phony_dim_3" in dims:
                new_dims = tuple(dim for dim in dims if dim != "phony_dim_3")
                new_shape = tuple(var_data.sizes[dim] for dim in new_dims)
                var_data = xr.DataArray(
                    var_data.values.reshape(new_shape),
                    dims=new_dims,
                    coords={dim: var_data.coords[dim] for dim in new_dims},
                )

            var_data = var_data.rename({"phony_dim_0": "ID"})

            new_vars = reformat_specific(name, var_name, var_data, new_vars)

    return xr.Dataset(new_vars)


def cses_to_unix(cses_time):
    """
    Converts CSES time to UNIX time.
    :param cses_time: The CSES time to convert (can also be a datetime object or a DataArray object with a CSES time)
    :return: The UNIX time
    """
    if int(cses_time) <= 0:
        return 0
    if type(cses_time) is np.ndarray and len(cses_time) == 1:
        cses_time = cses_time[0]

    if type(cses_time) in [int, np.int64, np.int32, np.int16, np.int8]:
        cses_time = str(cses_time)

    if type(cses_time) is xr.DataArray:
        cses_time = str(cses_time.values)
        cses_datetime = datetime.datetime(
            int(cses_time[0:4]),
            int(cses_time[4:6]),
            int(cses_time[6:8]),
            int(cses_time[8:10]),
            int(cses_time[10:12]),
            int(cses_time[12:14]),
        )
    elif type(cses_time) is str:
        cses_datetime = datetime.datetime(
            int(cses_time[0:4]),
            int(cses_time[4:6]),
            int(cses_time[6:8]),
            int(cses_time[8:10]),
            int(cses_time[10:12]),
            int(cses_time[12:14]),
        )
    elif type(cses_time) is datetime.datetime:
        cses_datetime = cses_time
    else:
        raise TypeError(
            "cses_time must be either a string, an xarray DataArray or a datetime object."
        )

    return int(cses_datetime.timestamp())


def normalize(
    filepath: str,
    time_col="UTC_TIME",
    time_type="CSES",
    geo_lat_col="GEO_LAT",
    geo_long_col="GEO_LON",
):
    """
    Removes artifacts from file and re-exports it as a .csv file.
    First columns are time (UNIX), geographic latitude, geographic longitude, and the rest are the feature columns.
    :param filepath: The path of the file to import (has to be a .h5 file)
    :param time_col: The name of the time column. (default: 'UTC_TIME')
    :param time_type: Time type, can be 'CSES' (As defined in the CSES-1 Manual) or 'UNIX'. (default: 'CSES')
    :param geo_lat_col: The name of the geographic latitude column. (default: 'GEO_LAT')
    :param geo_long_col: The name of the geographic longitude column. (default: 'GEO_LONG')
    :return: xarray Dataset
    """
    if not filepath.endswith(".h5") and not filepath.endswith(".zarr.zip"):
        raise ValueError("File path must be an .h5 file or a .zarr.zip file.")

    ds = xr.open_zarr(filepath)

    ds = reformat_xr(ds)

    if time_type not in ["CSES", "UNIX"]:
        raise ValueError('time must be either "CSES" or "UNIX"')

    if time_type == "CSES":
        ds[time_col] = ("ID", [cses_to_unix(i) for i in ds[time_col].values])

    ds = ds.rename({time_col: "TIME", geo_lat_col: "GEO_LAT", geo_long_col: "GEO_LON"})
    ds = ds.sortby("TIME")

    return ds


def group_by_time(file, timeseries=60, by="mean"):
    """
    Groups a Dataset by time:
    :param file: The dataset to group (can be either a filepath or a Dataset). Dataset has to be normalized with the normalizer function
    :param timeseries: The time to group by, in seconds (default: 60))
    :param by: The method to group by. Can be 'mean', 'sum', 'min', 'max', 'first', 'last' or a function (default: 'mean')
    :return: The grouped Dataset
    """
    if type(file) is str:
        ds = xr.open_dataset(file)
    elif type(file) is xr.Dataset:
        ds = file
    else:
        raise TypeError("file must be either a string or a xarray Dataset.")

    if "TIME" not in ds:
        raise ValueError('Dataset must have a "TIME" column.')

    ds = ds.sortby("TIME")

    _, times = np.unique((ds["TIME"].values // timeseries), return_index=True)

    new_ds = xr.Dataset()
    for k in ds.data_vars:
        j = times[1]
        if len(ds[k].values) > 0:
            temp = np.zeros(len(times) - 2)
            for p, i in enumerate(times[2:]):
                if k == "TIME":
                    temp[p] = int(ds[k].squeeze().values[j])
                elif by == "mean":
                    temp[p] = ds[k].squeeze().values[j:i].mean()
                elif by == "sum":
                    temp[p] = ds[k].squeeze().values[j:i].sum()
                elif by == "min":
                    temp[p] = ds[k].squeeze().values[j:i].min()
                elif by == "max":
                    temp[p] = ds[k].squeeze().values[j:i].max()
                elif by == "first":
                    temp[p] = ds[k].squeeze().values[j]
                elif by == "last":
                    temp[p] = ds[k].squeeze().values[i]
                elif callable(by):
                    temp[p] = by(ds[k].squeeze().values[j:i])
                else:
                    raise ValueError(
                        'by must be either "mean", "sum", "min", "max","first", "last" or a function.'
                    )
                j = i
            new_ds[k] = xr.DataArray.from_dict({"dims": "ID", "data": temp})
    new_ds = new_ds.sortby("TIME")
    new_ds["ID"] = np.arange(len(new_ds["TIME"]))

    return new_ds


def merge_datasets(files: list):
    """
    Merges multiple datasets into one.
    :param files: The list of files to merge (can be a list of datasets or a list of filepaths, datasets must be normalized)
    :return: The merged Dataset
    """
    if len(files) < 2:
        raise ValueError("List of files must include at least 2 elements.")

    new_files = []
    if any([type(file) is np.str_ for file in files]):

        for index, i in enumerate(tqdm(files, desc="Normalizing files")):
            new_i = normalize(i)
            if "HEPP_X" in i:
                # the first 100 values are squewed and the last 10 are also squewed
                new_i = new_i.isel(ID=slice(100, len(new_i.ID) - 10))

            if "HEPP_L" in i:
                # the first 10 values are squewed sometimes
                new_i = new_i.isel(ID=slice(10, len(new_i.ID)))

            if "LAP" in i:
                # remove all values below 0K and above 4500K
                new_i = new_i.where(new_i["A321"] > 0, drop=True)
                new_i = new_i.where(new_i["A321"] < 4500, drop=True)

            if len(new_i["ID"].values) < 60:
                continue

            new_i = group_by_time(new_i)

            new_i = GlobeMap(new_i, 10, 10)

            new_files.append(new_i)

    elif not [file for file in files if type(file) is xr.Dataset]:
        raise ValueError(
            "files must be a list of xarray Datasets or a list of filenames."
        )

    ds = new_files[0]
    for file in tqdm(new_files[1:], desc="Merging datasets"):
        if type(file) is xr.Dataset:
            ds.coords["ID"] = np.arange(len(ds.coords["ID"].values))
            file.coords["ID"] = np.arange(
                len(ds.coords["ID"].values),
                len(ds.coords["ID"].values) + len(file.coords["ID"].values),
            )

            ds = xr.merge([ds, file])

    ds = ds.sortby("TIME")
    ds.coords["ID"] = np.arange(len(ds["ID"].values))

    return ds


def write_log(*text):
    with open(f"log_normalizer.txt", "a") as f:
        f.write(
            "[" + str(datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")) + "]  "
        )
        for x in text:
            f.write(str(x))
        f.write("\n")


def descreteLatLong(lat: float, long: float, lat_kern: float, long_kern: float):
    """
    This function will take a latitude and longitude and will return the block number in the grid.

    ### Parameters:
        -  lat: latitude
        -  long: longitude
        -  lat_kern: latitude resolution of the grid (lat_kern in deg)
        -  long_kern: longitude resolution of the grid (long_kernin deg)

    """

    x_block = int((long + 180) // long_kern)
    y_block = int((lat + 90) // lat_kern)

    return x_block, y_block


def GlobeMap(arr, lat_rap: int, long_rap: int, savetofile: str = None):
    """
    This function will take an xarray dataset and will return a new dataset with two new variables,
    Block_x and Block_y.

    ### Parameters:
      -  arr: xarray dataset
      -  lat_rap: latitude resolution of the grid (180/lat_rap)
      -  long_rap: longitude resolution of the grid (360/long_rap)
      -  savetofile: if you want to save the new dataset to a zarr file, provide the file name here. If not, the function will return the new dataset.


    """

    arr["Block_x"] = ({"ID": len(arr.ID)}, np.ones((len(arr.ID))))
    arr["Block_y"] = ({"ID": len(arr.ID)}, np.ones((len(arr.ID))))

    lat_kern = 180 / lat_rap
    long_kern = 360 / long_rap

    for i in range(len(arr.ID)):

        x_block, y_block = descreteLatLong(
            arr["GEO_LAT"][i].values.tolist(),
            arr["GEO_LON"][i].values.tolist(),
            lat_kern,
            long_kern,
        )

        arr["Block_x"][i] = x_block
        arr["Block_y"][i] = y_block

    if savetofile is not None:
        compressor = Blosc(cname="zstd", clevel=9, shuffle=Blosc.AUTOSHUFFLE)

        arr.to_zarr(
            savetofile,
            mode="w",
            encoding={k: {"compressor": compressor} for k in arr.data_vars},
        )

    return arr


def intersect_timestamps(directory):
    filelist = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            filelist.append(os.path.join(root, f))

    # Open all datasets
    datasets = [xr.open_zarr(file) for file in filelist]

    # Find the common timestamps
    common_timestamps = set(datasets[0]["TIME"].values.astype(int))
    for ds in datasets[1:]:
        common_timestamps &= set(ds["TIME"].values.astype(int))

    common_timestamps = sorted(common_timestamps)

    # Filter each dataset to include only the common timestamps
    filtered_datasets = []
    for ds in datasets:
        time_values = ds["TIME"].values.astype(int)
        mask = xr.DataArray(np.isin(time_values, common_timestamps), dims="ID")
        filtered_ds = ds.where(mask, drop=True)
        filtered_ds.coords["ID"] = np.arange(len(filtered_ds.coords["ID"].values))

        filtered_datasets.append(filtered_ds)

    return filtered_datasets


def process_payload(payload):
    files_list = np.load(f"working_files_{payload}.npy")
    merge_datasets(files_list).to_zarr(
        f"merged/{payload}.zarr.zip", mode="w"
    )
    filtered_datasets = intersect_timestamps("merged/")
    for _, _, filenames in os.walk("merged/"):
        for index, ds in enumerate(filtered_datasets):
            ds.to_zarr(
                "merged_filtered/"
                + filenames[index].split(".")[0]
                + ".zarr.zip",
                mode="w",
            )
    print(f"{payload} done")
