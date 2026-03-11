import random

import torch
import numpy as np
import torch.utils.data as data
import copy

import yaml
from easydict import EasyDict

from .utils.io import IO
from .utils.build import DATASETS
from .utils.logger import *
from .utils.build import build_dataset_from_cfg
from .utils.data import random_rotate_z, normalize_pc, augment_pc
import json
from tqdm import tqdm
import pickle
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
    
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)

    img = Image.open(path)
    return img.convert('RGB')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data

import os, sys, h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import torchvision.transforms as transforms

@DATASETS.register_module()
class ModelNet40_openshape(data.Dataset):
    def __init__(self, config):
        self.npoints = config.npoints
        self.process_data = True
        self.uniform = True
        self.generate_from_raw_data = False
        split = config.subset
        self.subset = config.subset
        self.openshape_setting = config.openshape_setting
        self.data_path = config.DATA_PATH
        self.catfile = os.path.join(self.data_path, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        self.pcs = np.load('%s/test_pc.npy' % self.data_path, allow_pickle=True)

        self.openshape_split = json.load(open('%s/test_split.json' % self.data_path, "r"))
        print_log('The size of %s data is %d' % (split, len(self.openshape_split)), logger='ModelNet')

        self.shape_names_addr = os.path.join(self.data_path, 'modelnet40_shape_names.txt')
        self.cate_to_id = {}
        with open(self.shape_names_addr) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        for i in range(len(lines)):
            self.cate_to_id[lines[i]] = str(i)

    def __len__(self):
        return len(self.openshape_split)

    def __getitem__(self, index):
        pc = copy.deepcopy(self.pcs[index])

        xyz = pc['xyz']
        rgb = pc['rgb'] 
        rgb = rgb / 255.0 # 100, scale to 0.4 to make it consistent with the training data
        rgb = torch.from_numpy(rgb).float()
        
        if self.openshape_setting:
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
            logging.info('flip yz')
            xyz = normalize_pc(xyz)
        else:
            xyz[:, 0:3] = pc_normalize(xyz[:, 0:3])
        xyz = torch.from_numpy(xyz).float()

        label_name = self.openshape_split[index]["category"]
        label = np.array([int(self.cate_to_id[label_name])]).astype(np.int32)

        return xyz, label[0], label_name, rgb
    

@DATASETS.register_module()
class ScanObjNN_openshape(data.Dataset):
    def __init__(self, config):
        self.npoints = config.npoints
        self.process_data = True
        self.uniform = True
        self.generate_from_raw_data = False
        self.openshape_setting = config.openshape_setting
        self.data_path = config.DATA_PATH

        self.categories = ["bag", "bin", "box", "cabinet", "chair", "desk", "display", "door", "shelf", "table", "bed", "pillow", "sink", "sofa", "toilet"]

        self.openshape_data = np.load('%s/xyz_label.npy' % self.data_path, allow_pickle=True).item()
        
        print_log('The size of Scanobjnn data is %d' % (len(self.openshape_data['xyz'])), logger='ScanObjNN')


    def __len__(self):
        return len(self.openshape_data['xyz'])
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, index):
        pc = copy.deepcopy(self.openshape_data['xyz'][index])

        xyz = pc

        if 'rgb' not in self.openshape_data:
            rgb = np.ones_like(xyz) * 0.4
        else:
            rgb = self.openshape_data['rgb'][index]

        if self.openshape_setting:
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
            logging.info('flip yz')
            xyz = normalize_pc(xyz)
        else:
            xyz = self.pc_norm(xyz)

        xyz = torch.from_numpy(xyz).float()
        rgb = torch.from_numpy(rgb).float()

        label = self.openshape_data['label'][index]
        label_name = self.categories[label]
        label = label.astype(np.int32)

        return xyz, label, label_name, rgb
  
# Embedding eva-E Dataloaders
@DATASETS.register_module()
class Ensembled_embedding(data.Dataset):
    def __init__(self, config):

        self.subset = config.subset
        self.npoints = config.npoints
        self.tokenizer = config.tokenizer
        self.train_transform = config.train_transform
        self.picked_rotation_degrees = list(range(12))  # 支持12个视角
        self.use_lvis = config.use_lvis
        self.text_source = ["text", "caption", "retrieval_text"] 
        self.pc_root = config.PC_PATH_ROOT
        
        # 图像特征是否在点云文件中（新格式）
        self.image_in_pc_file = getattr(config, 'IMAGE_IN_PC_FILE', False)
        self.image_feat_key = getattr(config, 'IMAGE_FEAT_KEY', 'image_feat')  # 图像特征字段名
        self.num_views = getattr(config, 'NUM_VIEWS', 12)  # 视角数量
        
        # 数据路径是否为绝对路径（不需要和 pc_root 拼接）
        self.use_absolute_path = getattr(config, 'USE_ABSOLUTE_PATH', False)
        
        # 如果图像不在点云文件中，则使用传统方式加载
        if not self.image_in_pc_file:
            self.image_root = config.IMAGE_PATH_ROOT
            image_data_ours_p = config.IMAGE_PATH
            with open(image_data_ours_p, 'r') as f:
                self.image_data_ours = json.load(f)
        else:
            self.image_root = None
            self.image_data_ours = None

        self.rgb_random_drop_prob = 0.5
            
        if self.use_lvis:
            logging.info("Using LVIS")
            self.data_list_file_openshape = config.PC_PATH_LIVS
        else:
            self.data_list_file_openshape = config.PC_PATH

        with open(self.data_list_file_openshape, 'r') as f:
            self.data_list_openshape = json.load(f)

        # GPT过滤文件（可选）
        gpt_filter_path = getattr(config, 'GPT_FILTER', None)
        if gpt_filter_path and os.path.exists(gpt_filter_path):
            self.gpt4_filtering = json.load(open(gpt_filter_path, "r"))
            self.use_text_filtering = True
        else:
            self.gpt4_filtering = {}
            self.use_text_filtering = False

        self.sample_points_num = self.npoints
        self.whole = config.get('whole')    # use both train and test data for pretraining

        self.file_list = []
        for key in self.data_list_openshape.keys():
            self.file_list.append({
                'model_id': key,
                'data_path': self.data_list_openshape[key]
            })
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger='Objaverse')

        # exit()

        self.permutation = np.arange(self.npoints)

        self.uniform = False
        self.augment = True
        # =================================================
        # TODO: disable for backbones except for PointNEXT!!!
        self.use_height = config.use_height
        # =================================================

        if self.augment:
            print("using augmented point clouds.")

        # self.template = "a point cloud model of {}."

        # generate random text
        self.default_text = self.generate_random_text(1024)
    
    def generate_random_text(self, embedding_dim=1024):
        random_array = np.random.random(embedding_dim)
        normalized_array = np.linalg.norm(random_array)
        normalized_array = random_array / normalized_array
        return normalized_array
    
    def generate_random_image_feat(self, embedding_dim=1024):
        """生成随机图像特征向量（用于缺失图像特征的情况）"""
        random_array = np.random.random(embedding_dim)
        normalized_array = random_array / np.linalg.norm(random_array)
        return normalized_array
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        if num < pc.shape[0]:
            np.random.shuffle(self.permutation)
            pc = pc[self.permutation[:num]]
        else:
            ran_sel = np.random.randint(0, pc.shape[0], num)
            pc = pc[ran_sel]
        return pc

    def __getitem__(self, idx):
        """加载单个样本，支持数据损坏时自动加载下一条"""
        max_attempts = 10  # 最多尝试加载 10 条不同的数据
        original_idx = idx
        
        for attempt in range(max_attempts):
            try:
                return self._load_sample(idx)
            except Exception as e:
                print(f"[WARNING] Failed to load sample {idx}: {str(e)}. Trying next sample (attempt {attempt + 1}/{max_attempts})...")
                # 尝试加载下一条数据，循环到数据集开头
                idx = (idx + 1) % len(self.file_list)
                # 如果绕回原点，说明尝试了很多次都失败
                if idx == original_idx:
                    break
        
        # 如果所有尝试都失败，返回 None（极端情况）
        print(f"[ERROR] Failed to load any valid sample after {max_attempts} attempts starting from index {original_idx}")
        return None
    
    def _load_sample(self, idx):
        """实际加载样本的内部方法"""
        sample = self.file_list[idx]

        name, data_path = sample['model_id'], sample['data_path']
        # 根据配置决定是否拼接根目录
        if self.use_absolute_path:
            openshape_path = data_path  # 直接使用绝对路径
        else:
            openshape_path = self.pc_root + data_path  # 拼接根目录
        
        max_retries = 3  # 最大重试次数
        for retry in range(max_retries):
            try:
                openshape_data = np.load(openshape_path, allow_pickle=True).item()
                data = openshape_data['xyz'].astype(np.float32)
                rgb = openshape_data['rgb'].astype(np.float32)
                break
            except (OSError, IOError) as e:
                # 临时性错误（如网络文件系统抖动），进行重试
                if retry < max_retries - 1:
                    print(f"Catched exception: {str(e)}. Re-trying ({retry + 1}/{max_retries})...")
                    import time
                    time.sleep(0.2)
                else:
                    raise RuntimeError(f"Failed to load {openshape_path} after {max_retries} retries: {e}")
            except Exception as e:
                # 永久性错误（如 pickle 损坏、数据格式错误），直接跳过不重试
                raise RuntimeError(f"Corrupted data file {openshape_path}: {e}")

        data = self.pc_norm(data)

        if self.augment:
            # data = random_point_dropout(data[None, ...]) #TODO to keep rgb correct
            data = random_scale_point_cloud(data[None, ...])
            data = shift_point_cloud(data)
            data = rotate_perturbation_point_cloud(data)
            data = rotate_point_cloud(data)
            data = data.squeeze()


        data = torch.from_numpy(data).float()

        if np.random.rand() < self.rgb_random_drop_prob:
            rgb = torch.from_numpy(rgb).float()
        else:
            rgb = torch.ones_like(data).float() * 0.4

        texts =[]
        if 'text' in self.text_source:
            if '-Objaverse' in data_path:
                if not (self.use_text_filtering and self.gpt4_filtering[name]["flag"] == "N"):
                    try:
                        data_text = openshape_data["text_feat"][0]
                        if(not isinstance(data_text,str)):
                            if np.random.rand() < 0.5:
                                texts.append(data_text['original'])
                            else:
                                texts.append(data_text['prompt_avg'])
                    except:
                        texts.append(self.default_text)

            else:
                idx = np.random.randint(len(openshape_data["text_feat"]))
                try:
                    data_text = openshape_data["text_feat"][idx]
                    if(not isinstance(data_text,str)):
                        if np.random.rand() < 0.5:
                            texts.append(data_text['original'])
                        else:
                            texts.append(data_text['prompt_avg'])
                except:
                    texts.append(self.default_text)
        
        if 'caption' in self.text_source:
            if np.random.rand() < 0.5:
                if len(openshape_data["blip_caption_feat"]) > 0:
                    try:
                        data_text = openshape_data["blip_caption_feat"]
                        if(not isinstance(data_text,str)):
                            if np.random.rand() < 0.5:
                                texts.append(data_text['original'])
                            else:
                                texts.append(data_text['prompt_avg'])
                            
                    except:
                        texts.append(self.default_text)
            else:
                if len(openshape_data["msft_caption_feat"]) > 0:
                    try:
                        data_text = openshape_data["msft_caption_feat"]
                        if(not isinstance(data_text,str)):
                            if np.random.rand() < 0.5:
                                texts.append(data_text['original'])
                            else:
                                texts.append(data_text['prompt_avg'])
                    except:
                        texts.append(self.default_text)
        
        if 'retrieval_text' in self.text_source:
            if len(openshape_data["retrieval_text"]) > 0:
                idx = np.random.randint(len(openshape_data["retrieval_text"]))
                try:
                    data_text = openshape_data["retrieval_text_feat"][idx]
                    if(not isinstance(data_text,str)):
                        texts.append(data_text['original'])

                except:
                    texts.append(self.default_text)

        if len(texts) > 0:
            text_idx = np.random.randint(len(texts))
            texts = texts[text_idx]
        else:
            texts = self.default_text


        # 加载图像特征
        if self.image_in_pc_file:
            # 新格式：图像特征在点云文件中的 image_feat 字段
            try:
                if self.image_feat_key in openshape_data:
                    image_feats = openshape_data[self.image_feat_key]  # [num_views, feat_dim]
                    sel_ind = random.choice(range(min(self.num_views, len(image_feats))))
                    image = image_feats[sel_ind]  # 选择一个视角
                    use_image = torch.tensor([1])
                else:
                    # 如果没有图像特征，使用随机向量
                    image = self.generate_random_image_feat()
                    use_image = torch.tensor([0])
            except Exception as e:
                print(f"Error loading image feat for {name}: {e}")
                image = self.generate_random_image_feat()
                use_image = torch.tensor([0])
        else:
            # 原始格式：图像特征单独存放
            try:
                image_path = self.image_data_ours[name]
                sel_ind = random.choice(self.picked_rotation_degrees)
                picked_image_addr = self.image_root + image_path[sel_ind] + '.npy'
                image = np.load(picked_image_addr)
                use_image = torch.tensor([1])
            except:
                image_path = self.image_data_ours['b1c821055c19413691ee708c3e2180a0']
                sel_ind = random.choice(self.picked_rotation_degrees)
                picked_image_addr = self.image_root + image_path[sel_ind] + '.npy'
                image = np.load(picked_image_addr)
                use_image = torch.tensor([0])

        texts = torch.from_numpy(texts)
        image = torch.from_numpy(image) if isinstance(image, np.ndarray) else image
        
        return name, name, use_image, texts, data, image, rgb
    
    def __len__(self):
        return len(self.file_list)

@DATASETS.register_module()
class Objaverse_lvis_openshape(data.Dataset):
    def __init__(self, config):

        self.subset = config.subset
        self.npoints = config.npoints
        self.tokenizer = config.tokenizer
        self.train_transform = config.train_transform
        self.picked_rotation_degrees = list(range(10))
        self.openshape_setting = config.openshape_setting
        self.prompt_template_addr = os.path.join('./data/templates.json')
        with open(self.prompt_template_addr) as f:
            self.templates = json.load(f)[config.pretrain_dataset_prompt]


        self.data_list_file = config.PC_PATH
        self.pc_root = config.PC_PATH_ROOT

        self.sample_points_num = self.npoints
        self.whole = config.get('whole')    # use both train and test data for pretraining

        # ============ 图像和文本特征相关配置（参考 Ensembled_embedding） ============
        self.text_source = getattr(config, 'text_source', ["text", "caption", "retrieval_text"])
        self.rgb_random_drop_prob = getattr(config, 'rgb_random_drop_prob', 0.5)
        
        # 图像特征是否在点云文件中（新格式）
        self.image_in_pc_file = getattr(config, 'IMAGE_IN_PC_FILE', True)  # 默认从点云文件中读取
        self.image_feat_key = getattr(config, 'IMAGE_FEAT_KEY', 'image_feat')  # 图像特征字段名
        self.num_views = getattr(config, 'NUM_VIEWS', 12)  # 视角数量
        
        # 如果图像不在点云文件中，则使用传统方式加载
        if not self.image_in_pc_file:
            self.image_root = getattr(config, 'IMAGE_PATH_ROOT', None)
            image_data_path = getattr(config, 'IMAGE_PATH', None)
            if image_data_path and os.path.exists(image_data_path):
                with open(image_data_path, 'r') as f:
                    self.image_data_ours = json.load(f)
            else:
                self.image_data_ours = None
        else:
            self.image_root = None
            self.image_data_ours = None
        
        # GPT过滤文件（可选）
        gpt_filter_path = getattr(config, 'GPT_FILTER', None)
        if gpt_filter_path and os.path.exists(gpt_filter_path):
            self.gpt4_filtering = json.load(open(gpt_filter_path, "r"))
            self.use_text_filtering = True
        else:
            self.gpt4_filtering = {}
            self.use_text_filtering = False
        
        # 生成默认随机文本特征
        self.default_text = self.generate_random_text(1024)

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='Objaverse')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger='Objaverse')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()

        self.file_list = []
        for line in lines:
            line = line.strip()
            point_path_relative = line.split(',')[3].replace('\n', '')
            # 从 point_path 中提取分组数字（"-" 后面的前三位整数）
            # point_path_relative 格式示例: "000-001/abc123/abc123.npz"
            # split('-')[1][:3] 得到 "001"，即 "-" 后面的三位数字
            group_id = point_path_relative.split('-')[1][:3]  # 提取 "001"
            # 将根目录中的 xxx 替换为当前分组数字，再拼接相对路径
            pc_root_with_group = self.pc_root.replace('xxx', group_id)
            self.file_list.append({                         
                'cate_id': line.split(',')[0],
                'cate_name': line.split(',')[1],
                'model_id': line.split(',')[2],
                'point_path': pc_root_with_group + point_path_relative
            })
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger='Objaverse')

        # exit()

        self.permutation = np.arange(self.npoints)

        self.uniform = False
        self.augment = False
        # =================================================
        # TODO: disable for backbones except for PointNEXT!!!
        self.use_height = config.use_height
        # =================================================

        if self.augment:
            print("using augmented point clouds.")

        # self.template = "a point cloud model of {}."

    def generate_random_text(self, embedding_dim=1024):
        """生成随机文本特征向量（用于缺失文本特征的情况）"""
        random_array = np.random.random(embedding_dim)
        normalized_array = np.linalg.norm(random_array)
        normalized_array = random_array / normalized_array
        return normalized_array
    
    def generate_random_image_feat(self, embedding_dim=1024):
        """生成随机图像特征向量（用于缺失图像特征的情况）"""
        random_array = np.random.random(embedding_dim)
        normalized_array = random_array / np.linalg.norm(random_array)
        return normalized_array

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        if num < pc.shape[0]:
            np.random.shuffle(self.permutation)
            pc = pc[self.permutation[:num]]
        else:
            ran_sel = np.random.randint(0, pc.shape[0], num)
            pc = pc[ran_sel]
        return pc

    def __getitem__(self, idx):
        """加载单个样本，支持数据损坏时自动加载下一条"""
        max_attempts = 10  # 最多尝试加载 10 条不同的数据
        original_idx = idx
        
        for attempt in range(max_attempts):
            try:
                return self._load_sample(idx)
            except Exception as e:
                print(f"[WARNING] Failed to load sample {idx}: {str(e)}. Trying next sample (attempt {attempt + 1}/{max_attempts})...")
                # 尝试加载下一条数据，循环到数据集开头
                idx = (idx + 1) % len(self.file_list)
                # 如果绕回原点，说明尝试了很多次都失败
                if idx == original_idx:
                    break
        
        # 如果所有尝试都失败，返回 None（极端情况）
        print(f"[ERROR] Failed to load any valid sample after {max_attempts} attempts starting from index {original_idx}")
        return None
    
    def _load_sample(self, idx):
        """实际加载样本的内部方法"""
        sample = self.file_list[idx]

        cate_id, cate_name, model_id, point_path = sample['cate_id'], sample['cate_name'], sample['model_id'], sample['point_path']

        max_retries = 3  # 最大重试次数
        for retry in range(max_retries):
            try:
                openshape_data = np.load(point_path, allow_pickle=True).item()
                data = openshape_data['xyz'].astype(np.float32)
                rgb = openshape_data['rgb'].astype(np.float32)
                break
            except (OSError, IOError) as e:
                # 临时性错误（如网络文件系统抖动），进行重试
                if retry < max_retries - 1:
                    print(f"Catched exception: {str(e)}. Re-trying ({retry + 1}/{max_retries})...")
                    import time
                    time.sleep(0.5)
                else:
                    raise RuntimeError(f"Failed to load {point_path} after {max_retries} retries: {e}")
            except Exception as e:
                # 永久性错误（如 pickle 损坏、数据格式错误），直接跳过不重试
                raise RuntimeError(f"Corrupted data file {point_path}: {e}")

        if self.openshape_setting:
            data[:, [1, 2]] = data[:, [2, 1]]
            logging.info('flip yz')
            data = normalize_pc(data)
        else:
            data = self.pc_norm(data)
        if self.augment:
            data = random_point_dropout(data[None, ...])
            data = random_scale_point_cloud(data)
            data = shift_point_cloud(data)
            data = rotate_perturbation_point_cloud(data)
            data = rotate_point_cloud(data)
            data = data.squeeze()

        if self.use_height:
            self.gravity_dim = 1
            height_array = data[:, self.gravity_dim:self.gravity_dim + 1] - data[:,
                                                                       self.gravity_dim:self.gravity_dim + 1].min()
            data = np.concatenate((data, height_array), axis=1)
            data = torch.from_numpy(data).float()
        else:
            data = torch.from_numpy(data).float()

        # ============ 加载文本特征（参考 Ensembled_embedding） ============
        texts = []
        if 'text' in self.text_source:
            if not (self.use_text_filtering and model_id in self.gpt4_filtering and self.gpt4_filtering[model_id].get("flag") == "N"):
                try:
                    if "text_feat" in openshape_data and len(openshape_data["text_feat"]) > 0:
                        data_text = openshape_data["text_feat"][0]
                        if not isinstance(data_text, str):
                            if np.random.rand() < 0.5:
                                texts.append(data_text['original'])
                            else:
                                texts.append(data_text['prompt_avg'])
                except:
                    texts.append(self.default_text)
        
        if 'caption' in self.text_source:
            if np.random.rand() < 0.5:
                if "blip_caption_feat" in openshape_data and len(openshape_data["blip_caption_feat"]) > 0:
                    try:
                        data_text = openshape_data["blip_caption_feat"]
                        if not isinstance(data_text, str):
                            if np.random.rand() < 0.5:
                                texts.append(data_text['original'])
                            else:
                                texts.append(data_text['prompt_avg'])
                    except:
                        texts.append(self.default_text)
            else:
                if "msft_caption_feat" in openshape_data and len(openshape_data["msft_caption_feat"]) > 0:
                    try:
                        data_text = openshape_data["msft_caption_feat"]
                        if not isinstance(data_text, str):
                            if np.random.rand() < 0.5:
                                texts.append(data_text['original'])
                            else:
                                texts.append(data_text['prompt_avg'])
                    except:
                        texts.append(self.default_text)
        
        if 'retrieval_text' in self.text_source:
            if "retrieval_text_feat" in openshape_data and len(openshape_data.get("retrieval_text", [])) > 0:
                idx_text = np.random.randint(len(openshape_data["retrieval_text"]))
                try:
                    data_text = openshape_data["retrieval_text_feat"][idx_text]
                    if not isinstance(data_text, str):
                        texts.append(data_text['original'])
                except:
                    texts.append(self.default_text)

        if len(texts) > 0:
            text_idx = np.random.randint(len(texts))
            texts = texts[text_idx]
        else:
            texts = self.default_text

        # ============ 加载图像特征（参考 Ensembled_embedding） ============
        # print(f'image_in_pc_file: {self.image_in_pc_file}')
        if self.image_in_pc_file:
            # 新格式：图像特征在点云文件中的 image_feat 字段
            try:
                if self.image_feat_key in openshape_data:
                    image_feats = openshape_data[self.image_feat_key]  # [num_views, feat_dim]
                    sel_ind = random.choice(range(min(self.num_views, len(image_feats))))
                    image = image_feats[sel_ind]  # 选择一个视角
                    use_image = torch.tensor([1])
                else:
                    # 如果没有图像特征，使用随机向量
                    image = self.generate_random_image_feat()
                    use_image = torch.tensor([0])
            except Exception as e:
                print(f"Error loading image feat for {model_id}: {e}")
                image = self.generate_random_image_feat()
                use_image = torch.tensor([0])
        else:
            # 原始格式：图像特征单独存放
            try:
                if self.image_data_ours and model_id in self.image_data_ours:
                    image_path = self.image_data_ours[model_id]
                    sel_ind = random.choice(self.picked_rotation_degrees)
                    picked_image_addr = self.image_root + image_path[sel_ind] + '.npy'
                    image = np.load(picked_image_addr)
                    use_image = torch.tensor([1])
                else:
                    image = self.generate_random_image_feat()
                    use_image = torch.tensor([0])
            except:
                image = self.generate_random_image_feat()
                use_image = torch.tensor([0])

        # 转换为 tensor
        texts = torch.from_numpy(texts) if isinstance(texts, np.ndarray) else torch.tensor(texts)
        image = torch.from_numpy(image) if isinstance(image, np.ndarray) else image
        rgb = torch.from_numpy(rgb).float() if isinstance(rgb, np.ndarray) else rgb

        cate_id = np.array([cate_id]).astype(np.int32)
        
        # 返回格式与 Ensembled_embedding 一致：
        # (name, name, use_image, texts, data, image, rgb)
        # 这里 name 用 model_id，同时也返回 cate_name 用于分类
        return data, cate_id, cate_name, rgb, model_id, use_image, texts, image

    def __len__(self):
        return len(self.file_list)  
    

import collections.abc as container_abcs
int_classes = int
# from torch._six import string_classes
string_classes = str
import re
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")
np_str_obj_array_pattern = re.compile(r'[SaUO]')

def customized_collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size
    
    支持自动过滤损坏的数据（返回 None 的样本）
    """
    # 过滤掉 None 值（损坏的数据）
    if isinstance(batch, list):
        batch = [example for example in batch if example is not None]
        # 如果整个 batch 都是损坏数据，返回空
        if len(batch) == 0:
            return None
    
    elem = batch[0]
    elem_type = type(elem)

    # 额外检查：过滤掉 example[4] 为 None 的情况（点云数据为空）
    if isinstance(batch, list) and isinstance(elem, (tuple, list)) and len(elem) > 4:
        batch = [example for example in batch if example[4] is not None]

    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            # storage = elem.untyped_storage()._new_shared(numel)
            out = elem.new(storage)
            out.resize_(0)
        # import pdb; pdb.set_trace()
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return customized_collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: customized_collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(customized_collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [customized_collate_fn(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config

def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    merge_new_config(config=config, new_config=new_config)
    return config

class Dataset_3D():
    def __init__(self, args, tokenizer, dataset_type, train_transform=None):
        if dataset_type == 'train':
            self.dataset_name = args.pretrain_dataset_name
        elif dataset_type == 'val':
            self.dataset_name = args.validate_dataset_name
        elif dataset_type == 'val_lvis':
            self.dataset_name = args.validate_dataset_name_lvis
        elif dataset_type == 'val_scanobjnn':
            self.dataset_name = args.validate_dataset_name_scanobjnn
        else:
            raise ValueError("not supported dataset type.")
        with open('./data/dataset_catalog.json', 'r') as f:
            self.dataset_catalog = json.load(f)
            self.dataset_usage = self.dataset_catalog[self.dataset_name]['usage']
            self.dataset_split = self.dataset_catalog[self.dataset_name][self.dataset_usage]
            self.dataset_config_dir = self.dataset_catalog[self.dataset_name]['config']
        self.tokenizer = tokenizer
        self.train_transform = train_transform
        self.pretrain_dataset_prompt = args.pretrain_dataset_prompt
        self.validate_dataset_prompt = args.validate_dataset_prompt
        self.build_3d_dataset(args, self.dataset_config_dir)

    def build_3d_dataset(self, args, config):
        config = cfg_from_yaml_file(config)
        config.tokenizer = self.tokenizer
        config.train_transform = self.train_transform
        config.pretrain_dataset_prompt = self.pretrain_dataset_prompt
        config.validate_dataset_prompt = self.validate_dataset_prompt
        config.args = args
        config.use_height = args.use_height
        config.npoints = args.npoints
        config.openshape_setting = args.openshape_setting
        config.use_lvis = args.use_lvis
        config_others = EasyDict({'subset': self.dataset_split, 'whole': False})
        self.dataset = build_dataset_from_cfg(config, config_others)