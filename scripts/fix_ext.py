import os
from PIL import Image
from tqdm import tqdm


def fix_extensions(root_dir):
    for split in ["train", "test"]:
        split_path = os.path.join(root_dir, split)

        for class_dir in os.listdir(split_path):
            class_path = os.path.join(split_path, class_dir)

            if not os.path.isdir(class_path):
                continue

            for file in tqdm(os.listdir(class_path)):
                file_path = os.path.join(class_path, file)

                if "." in file:
                    continue

                try:
                    with Image.open(file_path) as img:
                        format = img.format.lower()  # jpeg / png

                    new_name = file + "." + format
                    new_path = os.path.join(class_path, new_name)

                    os.rename(file_path, new_path)

                except Exception as e:
                    print(f"Error with {file}: {e}")


if __name__ == "__main__":
    fix_extensions("data")