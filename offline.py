from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np
import os

if __name__ == '__main__':
    fe = FeatureExtractor()

    # added support for nested directories
    for img_path in (sorted(Path("./static/img").glob("**/?*.*"))):
        print(img_path)  # e.g., ./static/img/xxx.jpg
        feature = fe.extract(img=Image.open(img_path))

        # replace the static directory and keep the filename for reverse lookup later
        feature_path = Path(str(img_path).replace("static/img", "static/feature") + ".npy")
        # print(feature_path)  # e.g., ./static/feature/xxx.npy

        # create a folder if it doesnt exist
        parent_folder = feature_path.parent
        if not os.path.exists(parent_folder):
            os.makedirs(parent_folder)

        np.save(feature_path, feature)
