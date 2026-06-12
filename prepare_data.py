import glob, os
from Dataset.synthesize_data import synthsize

all_images = glob.glob('indoorCVPR_09/Images/**/*.jpg', recursive=True)

# 80/20 split
n_train = int(len(all_images) * 0.8)
train_imgs = all_images[:n_train]
eval_imgs  = all_images[n_train:]

os.makedirs('data/train/flk_img', exist_ok=True)
os.makedirs('data/train/org_img', exist_ok=True)
os.makedirs('data/eval/flk_img',  exist_ok=True)
os.makedirs('data/eval/org_img',  exist_ok=True)

synthsize(train_imgs, 'data/train/flk_img', 'data/train/org_img', model=0)  # Fluorescent
synthsize(eval_imgs,  'data/eval/flk_img',  'data/eval/org_img',  model=0)