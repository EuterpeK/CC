import PIL.Image
from PIL.ImageOps import exif_transpose
import os
import numpy as np

import torchvision.transforms as tvf
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def _resize_pil_image(img, long_edge_size, args):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    # new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    # new_size = tuple(max(new_size), max(new_size))
    # new_size = tuple((448, 224))
    new_size = tuple((args.width, args.height))
    return img.resize(new_size, interp)


def load_images(folder_or_list, args, square_ok=False):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        # print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    imgs = []
    for path in folder_content:
        if not path.endswith(('.jpg', '.jpeg', '.png', '.JPG')):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
        # W1, H1 = img.size
        # if size == 224:
        #     # resize short side to 224 (then crop)
        #     img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        # else:
        #     # resize long side to 512
        #     img = _resize_pil_image(img, size, args)
        
        long_edge_size = max(args.width, args.height)
        img = _resize_pil_image(img, long_edge_size, args)
        
        W, H = img.size
        cx, cy = W//2, H//2
        # if size == 224:
        #     half = min(cx, cy)
        #     img = img.crop((cx-half, cy-half, cx+half, cy+half))
        # else:
        #     halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
        #     if not (square_ok) and W == H:
        #         halfh = 3*halfw/4
        #     img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
        halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
        img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
        # W2, H2 = img.size
        # print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    assert imgs, 'no images foud at '+root
    # print(f' (Found {len(imgs)} images)')
    return imgs


# def _resize_pil_image(img, long_edge_size):
#     S = max(img.size)
#     if S > long_edge_size:
#         interp = PIL.Image.LANCZOS
#     elif S <= long_edge_size:
#         interp = PIL.Image.BICUBIC
#     new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
#     # new_size = tuple(max(new_size), max(new_size))
#     # new_size = tuple((448, 224))
#     return img.resize(new_size, interp)


# def load_images(folder_or_list, size, square_ok=False):
#     """ open and convert all images in a list or folder to proper input format for DUSt3R
#     """
#     if isinstance(folder_or_list, str):
#         # print(f'>> Loading images from {folder_or_list}')
#         root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

#     elif isinstance(folder_or_list, list):
#         # print(f'>> Loading a list of {len(folder_or_list)} images')
#         root, folder_content = '', folder_or_list

#     else:
#         raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

#     imgs = []
#     for path in folder_content:
#         if not path.endswith(('.jpg', '.jpeg', '.png', '.JPG')):
#             continue
#         img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
#         W1, H1 = img.size
#         if size == 224:
#             # resize short side to 224 (then crop)
#             img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
#         else:
#             # resize long side to 512
#             img = _resize_pil_image(img, size)
#         W, H = img.size
#         cx, cy = W//2, H//2
#         if size == 224:
#             half = min(cx, cy)
#             img = img.crop((cx-half, cy-half, cx+half, cy+half))
#         else:
#             halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
#             if not (square_ok) and W == H:
#                 halfh = 3*halfw/4
#             img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

#         W2, H2 = img.size
#         # print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
#         imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
#             [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

#     assert imgs, 'no images foud at '+root
#     # print(f' (Found {len(imgs)} images)')
#     return imgs
