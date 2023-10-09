import imghdr
import os
from pathlib import Path

train_image_path = "./train-data"


# only keep the files which are acceptable by tensorflow
for category in ["cat","dog"]:
    data_dir = os.path.join(train_image_path, category)
    image_extensions = [".png", ".jpg"]
    img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
    for filepath in Path(data_dir).rglob("*"):
        if filepath.suffix.lower() in image_extensions:
            img_type = imghdr.what(filepath)
            if img_type is None:
                print(f"{filepath} is not an image, so removing.")
                os.remove(filepath)
            elif img_type not in img_type_accepted_by_tf:
                print(f"{filepath} is a {img_type}, not accepted by TensorFlow, so removing.")
                os.remove(filepath)
