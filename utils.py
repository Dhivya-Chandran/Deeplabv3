import numpy as np
try:
    import cv2
except Exception:
    cv2 = None
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageOps, ImageEnhance
import torch
import glob


# ============ Data Augmentation Functions ============
def augment_pair(img_pil, mask_pil, aug_config=None):
    """
    Apply augmentation to image and mask pair (maintains consistency).
    
    Args:
        img_pil: PIL Image (RGB)
        mask_pil: PIL Image (grayscale segmentation mask)
        aug_config: dict with keys 'hflip' (bool), 'brightness' (factor), 'contrast' (factor)
    
    Returns:
        (augmented_img_pil, augmented_mask_pil)
    """
    if aug_config is None:
        aug_config = {}
    
    # Horizontal flip (applied to both)
    if aug_config.get('hflip', False):
        img_pil = ImageOps.mirror(img_pil)
        mask_pil = ImageOps.mirror(mask_pil)
    
    # Color augmentation (only on image)
    if aug_config.get('brightness', 1.0) != 1.0:
        enhancer = ImageEnhance.Brightness(img_pil)
        img_pil = enhancer.enhance(aug_config['brightness'])
    
    if aug_config.get('contrast', 1.0) != 1.0:
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(aug_config['contrast'])
    
    if aug_config.get('saturation', 1.0) != 1.0:
        enhancer = ImageEnhance.Color(img_pil)
        img_pil = enhancer.enhance(aug_config['saturation'])
    
    return img_pil, mask_pil


def get_augmentation_config(prob_hflip=0.5, brightness_range=(0.8, 1.2), 
                            contrast_range=(0.8, 1.2), saturation_range=(0.8, 1.2)):
    """
    Generate random augmentation configuration.
    
    Returns:
        dict with augmentation parameters
    """
    config = {
        'hflip': np.random.rand() < prob_hflip,
        'brightness': np.random.uniform(*brightness_range),
        'contrast': np.random.uniform(*contrast_range),
        'saturation': np.random.uniform(*saturation_range),
    }
    return config

def create_class_mask(img, color_map, is_normalized_img=True, is_normalized_map=False, show_masks=False):
    """
    Function to create C matrices from the segmented image, where each of the C matrices is for one class
    with all ones at the pixel positions where that class is present

    img = The segmented image

    color_map = A list with tuples that contains all the RGB values for each color that represents
                some class in that image

    is_normalized_img = Boolean - Whether the image is normalized or not
                        If normalized, then the image is multiplied with 255

    is_normalized_map = Boolean - Represents whether the color map is normalized or not, if so
                        then the color map values are multiplied with 255

    show_masks = Wherether to show the created masks or not
    """

    if is_normalized_img and (not is_normalized_map):
        img *= 255

    if is_normalized_map and (not is_normalized_img):
        img = img / 255
    
    mask = []
    hw_tuple = img.shape[:-1]
    for color in color_map:
        color_img = []
        for idx in range(3):
            color_img.append(np.ones(hw_tuple) * color[idx])

        color_img = np.array(color_img, dtype=np.uint8).transpose(1, 2, 0)

        mask.append(np.uint8((color_img == img).sum(axis = -1) == 3))

    return np.array(mask)


# Cityscapes dataset Loader

def loader_cscapes(input_path, segmented_path, batch_size, h=1024, w=2048, limited=False, augment=True):
    filenames_t = sorted(glob.glob(input_path + '/**/*.png', recursive=True), key=lambda x : int(os.path.basename(x).split('_')[1] + os.path.basename(x).split('_')[2]))
    total_files_t = len(filenames_t)
    
    filenames_s = sorted(glob.glob(segmented_path + '/**/*trainIds.png', recursive=True), key=lambda x : int(os.path.basename(x).split('_')[1] + os.path.basename(x).split('_')[2]))
    
    total_files_s = len(filenames_s)
    
    assert(total_files_t == total_files_s)
    
    batches = np.random.permutation(np.arange(total_files_s))
    idx0 = 0
    idx1 = idx0 + batch_size
    
    if str(batch_size).lower() == 'all':
        batch_size = total_files_s
    
    idx = 1 if not limited else total_files_s // batch_size + 1
    while(idx):
      
        batch = np.arange(idx0, idx1)
      
        # Choosing random indexes of images and labels
        batch_idxs = np.random.randint(0, total_files_s, batch_size)
        
        inputs = []
        labels = []
        
        for jj in batch_idxs:
            # Reading photo
            img = Image.open(filenames_t[jj])
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Reading semantic image
            mask = Image.open(filenames_s[jj])
            # Convert to grayscale if needed
            if mask.mode != 'L':
                mask = mask.convert('L')
            
            # Apply augmentation before resizing
            if augment:
                aug_config = get_augmentation_config()
                img, mask = augment_pair(img, mask, aug_config)
            
            # Resize using PIL (maintains quality)
            img = img.resize((w, h), Image.NEAREST)
            mask = mask.resize((w, h), Image.NEAREST)
            
            # Convert to array and normalize
            img = np.array(img, dtype=np.float32) / 255.0
            inputs.append(img)
          
            img = np.array(mask, dtype=np.int64)
            labels.append(img)
         
        inputs = np.stack(inputs, axis=0)
        # Changing image format to B x C x H x W
        inputs = torch.tensor(inputs, dtype=torch.float32).permute(0, 3, 1, 2)
        
        labels = np.stack(labels, axis=0)
        labels = torch.tensor(labels, dtype=torch.long)
        
        idx0 = idx1 if idx1 + batch_size < total_files_s else 0
        idx1 = idx0 + batch_size
        
        if limited:
          idx -= 1
          
        yield inputs, labels

def loader(training_path, segmented_path, batch_size, h=512, w=512, augment=True):
    """
    The Loader to generate inputs and labels from the Image and Segmented Directory

    Arguments:

    training_path - str - Path to the directory that contains the training images

    segmented_path - str - Path to the directory that contains the segmented images

    batch_size - int - the batch size
    
    augment - bool - whether to apply data augmentation

    yields inputs and labels of the batch size
    """

    filenames_t = os.listdir(training_path)
    total_files_t = len(filenames_t)
    
    filenames_s = os.listdir(segmented_path)
    total_files_s = len(filenames_s)
    
    assert(total_files_t == total_files_s)
    
    if str(batch_size).lower() == 'all':
        batch_size = total_files_s
    
    idx = 0
    while(1):
        batch_idxs = np.random.randint(0, total_files_s, batch_size)
            
        inputs = []
        labels = []
        
        for jj in batch_idxs:
            img = plt.imread(training_path + filenames_t[jj])
            if img.dtype != np.float32:
                img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0
            # Ensure 3 channels
            if len(img.shape) == 2:
                img = np.stack([img]*3, axis=2)
            # Use PIL for resizing to avoid OpenCV issues
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            
            # Load mask
            mask_pil = Image.open(segmented_path + filenames_s[jj])
            if mask_pil.mode != 'L':
                mask_pil = mask_pil.convert('L')
            
            # Apply augmentation before resizing
            if augment:
                aug_config = get_augmentation_config()
                img_pil, mask_pil = augment_pair(img_pil, mask_pil, aug_config)
            
            # Resize
            img_pil = img_pil.resize((w, h), Image.NEAREST)
            mask_pil = mask_pil.resize((w, h), Image.NEAREST)
            
            img = np.array(img_pil, dtype=np.float32) / 255.0
            inputs.append(img)
            
            img = np.array(mask_pil, dtype=np.int64)
            labels.append(img)
         
        inputs = np.stack(inputs, axis=0)
        inputs = torch.tensor(inputs, dtype=torch.float32).permute(0, 3, 1, 2)
        
        labels = np.stack(labels, axis=0)
        labels = torch.tensor(labels, dtype=torch.long)
        
        yield inputs, labels


def decode_segmap_camvid(image):
    Sky = [128, 128, 128]
    Building = [128, 0, 0]
    Pole = [192, 192, 128]
    Road_marking = [255, 69, 0]
    Road = [128, 64, 128]
    Pavement = [60, 40, 222]
    Tree = [128, 128, 0]
    SignSymbol = [192, 128, 128]
    Fence = [64, 64, 128]
    Car = [64, 0, 128]
    Pedestrian = [64, 64, 0]
    Bicyclist = [0, 128, 192]

    label_colors = np.array([Sky, Building, Pole, Road_marking, Road, 
                              Pavement, Tree, SignSymbol, Fence, Car, 
                              Pedestrian, Bicyclist]).astype(np.uint8)

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for label in range(len(label_colors)):
            r[image == label] = label_colors[label, 0]
            g[image == label] = label_colors[label, 1]
            b[image == label] = label_colors[label, 2]

    rgb = np.zeros((image.shape[0], image.shape[1], 3)).astype(np.uint8)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    return rgb

def decode_segmap_cscapes(image, nc=34):
    
    label_colours = np.array([(0, 0, 0),  # 0=background
                              (0, 0, 0),  # 1=ego vehicle
                              (0, 0, 0),  # 2=rectification border
                              (0, 0, 0),  # 3=out of toi
                              (0, 0, 0),  # 4=static
                 # 5=dynamic,    6=ground,       7=road,       8=sidewalk,    9=parking
                 (111, 74,  0),  ( 81,  0, 81), (128, 64,128), (244, 35,232), (250,170,160),
                 # 10=rail track, 11=building,     12=wall,       13=fence,       14=guard rail
                 (230,150,140),  ( 70, 70, 70), (102,102,156), (190,153,153), (180,165,180),
                 # 15=bridge,   16=tunnel,     17=pole,       18=pole group, 19=traffic light
                 (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30),
                 # 20=traffic sign, 21=vegetation, 22=terrain,    23=sky,        24=person
                 (220,220,  0),     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60),
                 # 25=rider,    26=car,        27=truck,      28=bus,        29=caravan, 
                 (255,  0,  0), (  0,  0,142), (  0,  0, 70), (  0, 60,100), (  0,  0, 90),
                 # 30=trailer,  31=train,        32=motorcycle, 33=bicycle,        34=license plate, 
                 (  0,  0,110), (  0, 80,100),   (  0,  0,230), (119, 11, 32), (  0,  0,142),
                 ])
    
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    
    for l in range(0, nc):
        r[image == l] = label_colours[l, 0]
        g[image == l] = label_colours[l, 1]
        b[image == l] = label_colours[l, 2]

    rgb = np.zeros((image.shape[0], image.shape[1], 3)).astype(np.uint8)
    rgb[:, :, 0] = b
    rgb[:, :, 1] = g
    rgb[:, :, 2] = r
    return rgb

def show_images(images, in_row=True):
    '''
    Helper function to show 3 images
    '''
    total_images = len(images)

    rc_tuple = (1, total_images)
    if not in_row:
        rc_tuple = (total_images, 1)
    
	#figure = plt.figure(figsize=(20, 10))
    for ii in range(len(images)):
        plt.subplot(*rc_tuple, ii+1)
        plt.title(images[ii][0])
        plt.axis('off')
        plt.imshow(images[ii][1])
    plt.show()

def get_class_weights(loader, num_classes, c=1.02):
    '''
    This class return the class weights for each class
    
    Arguments:
    - loader : The generator object which return all the labels at one iteration
               Do Note: That this class expects all the labels to be returned in
               one iteration

    - num_classes : The number of classes

    Return:
    - class_weights : An array equal in length to the number of classes
                      containing the class weights for each class
    '''

    _, labels = next(loader)
    all_labels = labels.flatten()
    each_class = np.bincount(all_labels, minlength=num_classes)
    prospensity_score = each_class / len(all_labels)
    class_weights = 1 / (np.log(c + prospensity_score))
    return class_weights
