import os
import shutil
import pandas as pd
from tqdm import tqdm


def prepare_split(csv_path, images_root, output_root):
    df = pd.read_csv(csv_path)

    os.makedirs(os.path.join(output_root, "venomous"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "non_venomous"), exist_ok=True)

    uuid_to_path = {}

    for subdir in os.listdir(images_root):
        subdir_path = os.path.join(images_root, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                filename_no_ext = os.path.splitext(file)[0]
                uuid_to_path[filename_no_ext] = os.path.join(subdir_path, file)


    for _, row in tqdm(df.iterrows(), total=len(df)):
        uuid = str(row["UUID"]).strip()
        label = row["poisonous"]

        if uuid not in uuid_to_path:
            continue

        src_path = uuid_to_path[uuid]

        if label == 1:
            dst_dir = os.path.join(output_root, "venomous")
        else:
            dst_dir = os.path.join(output_root, "non_venomous")

        shutil.copy(src_path, os.path.join(dst_dir, uuid))


if __name__ == "__main__":

    prepare_split(
        csv_path="archive/Csv/train.csv",
        images_root="archive/train",
        output_root="data/train"
    )

    prepare_split(
        csv_path="archive/Csv/test.csv",
        images_root="archive/test",
        output_root="data/test"
    )