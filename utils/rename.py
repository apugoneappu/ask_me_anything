import os

cwd = os.getcwd()
# print(cwd)
for idx, f in enumerate(os.listdir('./images/')):
    
    img_path = os.path.join(cwd, 'images', f)
    new_img_path = os.path.join(cwd, 'images', str(idx)+'.jpg')
    
    os.rename(img_path, new_img_path)

    if (not os.path.isfile(img_path)):
        print(img_path)

    feat_path = os.path.join(cwd, 'feats', f+'.npz')
    new_feat_path = os.path.join(cwd, 'feats', str(idx)+'.npz')


    if (not os.path.isfile(feat_path)):
        print(feat_path)

    os.rename(feat_path, new_feat_path)