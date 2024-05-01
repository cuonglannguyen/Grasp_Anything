import glob
import os
import re

import pickle
import torch
import pickle
import clip
from utils.dataset_processing import grasp, image, mask
from .grasp_data import GraspDatasetBase
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

class GraspAnythingPlusDataset(GraspDatasetBase):
 

    """
    Dataset wrapper for the Grasp-Anything dataset.
    """

    def __init__(self, file_path, ds_rotate=0, **kwargs):
        """
        :param file_path: Grasp-Anything Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(GraspAnythingPlusDataset, self).__init__(**kwargs)
        data_type = ''
        if kwargs['seen'] == True:
            data_type = 'seen'
        else:
            data_type = 'unseen'

        self.grasp_files = glob.glob(os.path.join(file_path, data_type, 'grasp_label', '*.pt'))
        self.prompt_files = glob.glob(os.path.join(file_path, data_type, 'grasp_instructions', '*.pkl'))
        self.rgb_files = glob.glob(os.path.join(file_path, data_type, 'image', '*.jpg'))
        #self.mask_files = glob.glob(os.path.join(file_path, 'mask', '*.npy'))
        
        '''
        if kwargs["seen"]:
            with open(os.path.join('split/grasp-anything/seen.obj'), 'rb') as f:
                idxs = pickle.load(f)

            self.grasp_files = list(filter(lambda x: x.split('/')[-1].split('.')[0] in idxs, self.grasp_files))
        else:
            with open(os.path.join('split/grasp-anything/unseen.obj'), 'rb') as f:
                idxs = pickle.load(f)

            self.grasp_files = list(filter(lambda x: x.split('/')[-1].split('.')[0] in idxs, self.grasp_files))
        '''
        self.grasp_files.sort()
        self.prompt_files.sort()
        self.rgb_files.sort()
        #self.mask_files.sort()

        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))
        self.model, self.preprocess = clip.load("ViT-B/32", device = self.device)

        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]
            

    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 416 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 416 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):       
        # Jacquard try
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx], scale=self.output_size / 416.0)

        c = self.output_size // 2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))

        # Cornell try
        # gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx])
        # center, left, top = self._get_crop_attrs(idx)
        # gtbbs.rotate(rot, center)
        # gtbbs.offset((-top, -left))
        # gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        depth_img = self.preprocess(torch.Tensor(depth_img.img)).unsqueeze(0).to(self.device)
        depth_features = self.model.encode_image(depth_img)
        return depth_features
    def normalize_image(self, img):
        # Convert the image to floating point data type
        img = img.convert('F')  # Convert the image to 32-bit floating point pixels

        # Normalize the image by scaling pixel values to range [0, 1]
        scale = 255.0
        img = Image.eval(img, lambda x: x / scale)

        # Apply manual normalization (e.g., [-1, 1] range)
        img = Image.eval(img, lambda x: (x - 0.5) / 0.5)

        # Convert back to RGB to maintain the output as a PIL image
        img = img.convert('RGB')
        
        return img
    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        #mask_file = self.grasp_files[idx].replace("positive_grasp", "mask").replace(".pt", ".npy")
        #mask_img = mask.Mask.from_file(mask_file)
        rgb_file = re.sub(r"_\d{1}\_\d{1}\.pt", ".jpg", self.grasp_files[idx])
        rgb_file = rgb_file.replace("grasp_label", "image")
        rgb_img = Image.open(rgb_file)
        #rgb_img = image.Image.mask_out_image(rgb_img, mask_img)

        # Jacquard try
        rgb_img.rotate(rot)
        width, height = rgb_img.size
        new_width = width * zoom
        new_height = height * zoom
        rgb_img = rgb_img.resize((int(new_width), int(new_height)), Image.BILINEAR)  # Apply zoom
        rgb_img = rgb_img.resize((self.output_size, self.output_size), Image.BILINEAR)  # Resize to output size

        #rgb_img.img = rgb_img.img.convert('RGB')

        if normalise:
            rgb_img = self.normalize_image(rgb_img)
        #with torch.no_grad():
        #    image_encode = self.preprocess(rgb_img).unsqueeze(0).to(self.device)
        #image_features = self.model.encode_image(image_encode)

        return rgb_img

        # Cornell try
        # center, left, top = self._get_crop_attrs(idx)
        # rgb_img.rotate(rot, center)
        # rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        # rgb_img.zoom(zoom)
        # rgb_img.resize((self.output_size, self.output_size))
        # if normalise:
        #     rgb_img.normalise()
        #     rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        # return rgb_img.img
    def get_text(self, idx):
        with open(self.prompt_files[idx], 'rb') as pickle_file:
            text = pickle.load(pickle_file)
            embed_text = clip.tokenize(text).to(self.device)
            with torch.no_grad():
               text_features = self.model.encode_text(embed_text)
            text_features =  text_features.detach().cpu().numpy()
            # Initialize and apply Gaussian Random Projection
            transformer = GaussianRandomProjection(n_components=224)
            reduced_data = transformer.fit_transform(text_features)

            # Step 3: Duplicate the reduced vector to create a 224x224x1 array
            expanded_array = np.tile(reduced_data, (224, 1))
            expanded_array = np.reshape(expanded_array, (224, 224, 1))  # Reshape to make it 224x224x1
        return expanded_array
