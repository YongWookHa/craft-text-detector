# Overview

Original CRAFT text detector's input image size is `384x384`. Though CRAFT showed good performance for scene text detection, the input size is not enough for the high-resolution task, especially when it comes to document.

This repository of CRAFT, you can change input image size for improving model performance at training.

# How to use

## Prepare your data

First of all, write your own Dataloader code.

In `datasets/craft_dataset.py`, you can find `CustomDataset`.

Make your `CustomDataset` return `image, char_boxes, words, image_fn` by `__getitem__` method. Return data format should be same as below.

- `image` : np.ndarray  
- `char_boxes` : character level bounding box coord.
    ```
    [   [lx, ly], [rx, ly], [rx, ry], [lx, ry],
        [lx, ly], [rx, ly], [rx, ry], [lx, ry],
        ...]   
    ```  
- `words` : list of words. character annotation should be in order of bounding boxes.
- `image_fn` : [pathlib](https://docs.python.org/3/library/pathlib.html) image path  

Then, change setting in `settings/default.yaml`.

```
train_data_path: <your train data path>
val_data_path: <your validate data path>
```  

These two setting is all you need to edit.  


Now you are ready to train your model. But the training might be very slow because of data processing time at making character and affinity heatmap.   

When it comes to train detecting text in high resolution documents, the heatmap processing is very slow.

In fact, the same data processing repeats every epoch. So, it does not necessarily have to be done for every epoch. Therefore, let's preprocess it before we start training.

Run `preprocess.py` like below.

```bash
python preprocess.py --setting settings/default.yaml --num_workers 16 --batch_size 4
```

## Train

```bash
python run.py --setting settings/default.yaml --version 0 --num_workers 16 -bs 4
```

To monitor the training progress, use tensorboard.

```bash
tensorboard --logdir tb_logs --bind_all
```