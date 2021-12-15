import torch
import cv2
import json
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms

from utils.craft_utils import load_image
from utils.data_manipulation import generate_affinity, generate_target


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir=None):
        super(CustomDataset, self).__init__()
        self.data_dir = data_dir
        self.json_names = []
        for json_fn in Path(data_dir).glob("**/*.json"):
            self.json_names.append(json_fn)

    def __len__(self):
        return len(self.json_names)

    def __getitem__(self, idx):
        fn = self.json_names[idx]
        image, char_boxes, words, image_fn = self.load_data(fn)

        return image, char_boxes, words, image_fn

    def load_data(self, json_fn: Path):
        """
        Write your own function here
        
        return:
            image, wordBoxes, list of words, image_fn
        """
        with json_fn.open('r', encoding='utf8') as f:
            data = json.load(f)
        image_fn = Path("{}".format(self.data_dir))
        image_fn = list(image_fn.glob("**/{}".format(data["image"]["file_name"])))

        if not image_fn:
            print("FileNotFoundError: image_fn : {}".format(image_fn))
            raise FileNotFoundError

        image = load_image(image_fn[0])

        char_boxes, words = [], []
        for word_d in data["text"]["word"]:
            text = word_d["value"]
            for d in word_d["letter"]:
                lx, ly, rx, ry = d["charbox"]
                char_boxes.append([[lx, ly], [rx, ly], [rx, ry], [lx, ry]])
            words.append(text)

        return image, np.array(char_boxes), words, image_fn[0]


class CustomCollate(object):
    def __init__(self, image_size, save_preprocessed=False):
        self.image_size = image_size
        self.save_preprocessed = save_preprocessed

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __call__(self, batch):
        """
        preprocess batch
        """
        batch_big_image, batch_weight_character, batch_weight_affinity = [], [], []

        if self.save_preprocessed:
            for image, char_boxes, words, fn in batch:
                char_boxes = np.transpose(char_boxes, (2, 1, 0))
                big_image, small_image, character = self.resize(image, char_boxes, big_side=self.image_size)  # Resize the image

                # Generate character heatmap
                weight_character = generate_target(small_image.shape, character.copy())

                # Generate affinity heatmap
                weight_affinity, _ = generate_affinity(small_image.shape, character.copy(), words)

                weight_character = weight_character.astype(np.float32)
                weight_affinity = weight_affinity.astype(np.float32)

                np.save(fn.parent / f"{fn.stem}_{self.image_size}_image.npy" , big_image)
                np.save(fn.parent / f"{fn.stem}_{self.image_size}_weight_character.npy", weight_character)
                np.save(fn.parent / f"{fn.stem}_{self.image_size}_weight_affinity.npy", weight_affinity)

                batch_big_image.append(self.image_transform(Image.fromarray(big_image)))
                batch_weight_character.append(weight_character)
                batch_weight_affinity.append(weight_affinity)
        else:
            for _, _, _, fn in batch:
                big_image = np.load(fn.parent / f"{fn.stem}_{self.image_size}_image.npy")
                weight_character = np.load(fn.parent / f"{fn.stem}_{self.image_size}_weight_character.npy")
                weight_affinity = np.load(fn.parent / f"{fn.stem}_{self.image_size}_weight_affinity.npy")

                batch_big_image.append(self.image_transform(Image.fromarray(big_image)))
                batch_weight_character.append(weight_character)
                batch_weight_affinity.append(weight_affinity)

        return  torch.stack(batch_big_image),  \
                torch.from_numpy(np.stack(batch_weight_character)),  \
                torch.from_numpy(np.stack(batch_weight_affinity))

    def resize(self, image, character, big_side):
        """
            Resizing the image while maintaining the aspect ratio and padding with average of the entire image to make the
            reshaped size = (side, side)
            :param image: np.array, dtype=np.uint8, shape=[height, width, 3]
            :param character: np.array, dtype=np.int32 or np.float32, shape = [2, 4, num_characters]
            :param side: new size to be reshaped to
            :return: resized_image, corresponding reshaped character bbox
        """

        height, width, channel = image.shape
        max_side = max(height, width)
        big_resize = (int(width/max_side*big_side), int(height/max_side*big_side))
        small_resize = (int(width/max_side*(big_side//2)), int(height/max_side*(big_side//2)))
        image = cv2.resize(image, big_resize)

        character = np.array(character)
        character[0, :, :] = character[0, :, :] * (small_resize[0] / width)
        character[1, :, :] = character[1, :, :] * (small_resize[1] / height)

        big_image = np.ones([big_side, big_side, 3], dtype=np.float32)*255
        h_pad, w_pad = (big_side-image.shape[0])//2, (big_side-image.shape[1])//2
        big_image[h_pad: h_pad + image.shape[0], w_pad: w_pad + image.shape[1]] = image
        big_image = big_image.astype(np.uint8)

        small_image = cv2.resize(big_image, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

        character[0, :, :] += (w_pad // 2)
        character[1, :, :] += (h_pad // 2)

        # character fit to small image
        return big_image, small_image, character

