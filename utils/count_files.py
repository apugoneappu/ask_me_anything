import os

def num_images() -> int:

    images = 0
    feats = 0

    for file in os.listdir('assets/images'):
        
        ext = file.split('.')[-1]
        if (ext == 'png' or ext == 'jpg' or ext == 'jpeg'):
            images += 1

    for file in os.listdir('assets/feats'):

        ext = file.split('.')[-1]
        if (ext == 'npz'):
            feats += 1

    assert images == feats, f'#images: {images} #feats: {feats}'
    return images

