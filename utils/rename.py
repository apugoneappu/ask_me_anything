import os
from shutil import copyfile

root_dir = '/Users/apoorve/Downloads/my'

dst_dir = '/Users/apoorve/Desktop/Personal/ama/assets'

src_images_dir = os.path.join(root_dir, 'images')
src_feat_dir = os.path.join(root_dir, 'feats')

dst_images_dir = os.path.join(dst_dir, 'images')
dst_feat_dir = os.path.join(dst_dir, 'feats')
# print(root_dir)

for idx, f in enumerate(os.listdir(src_images_dir)):
    
    img_path = os.path.join(src_images_dir, f)
    new_img_path = os.path.join(dst_images_dir, str(33+idx)+'.jpg')

    feat_path = os.path.join(src_feat_dir, f.split('.')[0] + '.npz')
    new_feat_path = os.path.join(dst_feat_dir, str(33+idx)+'.npz')

    if (os.path.isfile(img_path) and os.path.isfile(feat_path)):
        copyfile(img_path, new_img_path)
        copyfile(feat_path, new_feat_path)
        # os.rename(img_path, new_img_path)
        # os.rename(feat_path, new_feat_path)
    
    if (not os.path.isfile(img_path)):
        print(img_path)

    if (not os.path.isfile(feat_path)):
        print(feat_path)

