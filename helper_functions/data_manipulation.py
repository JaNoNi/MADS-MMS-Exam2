import logging
import os
import urllib.request
import zipfile

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


class DownloadProgressbar(tqdm):
    def update_to(self, block=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(block * block_size - self.n)


def create_activity_window(df_dict: dict[str, pd.DataFrame], agg_funcs: list, start: int, end: int):
    # Combine accelerometer and gyroscope data
    exp_types = list(df_dict.keys())
    first_type = exp_types.pop(0)
    df = pd.DataFrame(df_dict[first_type]).add_prefix(f"{first_type}_")
    for sub_type in exp_types:
        df = pd.merge(
            df,
            pd.DataFrame(df_dict[sub_type]).add_prefix(f"{sub_type}_"),
            left_index=True,
            right_index=True,
        )

    # Pivot wide
    df_stacked = df.iloc[start:end, :].agg(agg_funcs).stack().swaplevel()
    df_stacked.index = df_stacked.index.map("{0[1]}_{0[0]}".format)
    return df_stacked.to_frame().T


def download_raw_data(url: str, output_path: str):
    with DownloadProgressbar(unit="B", unit_scale=True, miniters=1, desc=output_path) as t_bar:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t_bar.update_to)


def open_raw_data(path_to_data: str, exp_type: str, exp_id: int, user_id: int) -> pd.DataFrame:
    filename = f"{exp_type}_exp{exp_id:02d}_user{user_id:02d}.txt"
    filedir = os.path.join(path_to_data, "RawData", filename)
    activity = pd.read_csv(filedir, sep=" ", header=None, names=["x", "y", "z"])
    return activity


def setup_raw_data(url: str, path_to_data: str, file_name: str = "RawData"):
    if os.path.isdir(path_to_data):
        if os.path.isfile(f"{path_to_data}/{file_name}"):
            logging.info("Found the correct data.")
            return
        else:
            logging.info(
                "Data folder found, but it does not contain the right data. Downloading..."
            )
    else:
        logging.info("Data folder not found. Downloading...")
        print("Data not found. Downloading...")

    os.makedirs(path_to_data, exist_ok=True)
    # Download raw data
    output_file = url.split("/")[-1]
    output_file_path = os.path.join(path_to_data, output_file)
    download_raw_data(url, output_file_path)
    # Unzip raw data
    if zipfile.is_zipfile(output_file_path):
        with zipfile.ZipFile(output_file_path, "r") as zip_ref:
            zip_ref.extractall(path_to_data)

    # Check if all files are present
    req_files = [file_name]
    logging.info("Checking if all files are present...")
    found_files = [_dir for _dir in os.listdir(path_to_data) if _dir in req_files]
    if len(found_files) == len(req_files):
        logging.info("All files are present")
        # Remove zip file
        if zipfile.is_zipfile(output_file_path):
            os.remove(output_file_path)
    else:
        raise FileNotFoundError(
            "Something went wrong. Please extract the zip file manually. "
            f"Your '{path_to_data}'-folder should contain the following files: {req_files}"
        )


def main():
    return ()


if __name__ == "__main__":
    main()
