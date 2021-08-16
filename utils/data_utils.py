import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

class DigitsDataset(Dataset):
    def __init__(self, data_path, channels, percent=0.1, filename=None, train=True, transform=None):
        if filename is None:
            if train:
                if percent >= 0.1:
                    for part in range(int(percent*10)):
                        if part == 0:
                            self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                        else:
                            images, labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                            self.images = np.concatenate([self.images,images], axis=0)
                            self.labels = np.concatenate([self.labels,labels], axis=0)
                else:
                    self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part0.pkl'), allow_pickle=True)
                    data_len = int(self.images.shape[0] * percent*10)
                    self.images = self.images[:data_len]
                    self.labels = self.labels[:data_len]
            else:
                self.images, self.labels = np.load(os.path.join(data_path, 'test.pkl'), allow_pickle=True)
        else:
            self.images, self.labels = np.load(os.path.join(data_path, filename), allow_pickle=True)

        self.transform = transform
        self.channels = channels
        self.labels = self.labels.astype(np.long).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.channels == 1:
            image = Image.fromarray(image, mode='L')
        elif self.channels == 3:
            image = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class OfficeDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load('../data/office_caltech_10_dataset/office_caltech_10/{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load('../data/office_caltech_10_dataset/office_caltech_10/{}_test.pkl'.format(site), allow_pickle=True)
            
        label_dict={'back_pack':0, 'bike':1, 'calculator':2, 'headphones':3, 'keyboard':4, 'laptop_computer':5, 'monitor':6, 'mouse':7, 'mug':8, 'projector':9}
        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class DomainNetDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if site == 'centralized':
            if train:
                self.paths, self.text_labels = np.load('../data/DomainNet/clipart_train.pkl', allow_pickle=True)
                for dataset in ['infograph', 'painting', 'quickdraw', 'real', 'sketch']:
                    temp_paths, _ = np.load('../data/DomainNet/{}_train.pkl'.format(dataset), allow_pickle=True)
                    np.append(self.paths, temp_paths)
            else:
                self.paths, self.text_labels = np.load('../data/DomainNet/clipart_test.pkl', allow_pickle=True)
                for dataset in ['infograph', 'painting', 'quickdraw', 'real', 'sketch']:
                    temp_paths, _ = np.load('../data/DomainNet/{}_test.pkl'.format(dataset), allow_pickle=True)
                    np.append(self.paths, temp_paths)
        else:
            if train:
                self.paths, self.text_labels = np.load('../data/DomainNet/{}_train.pkl'.format(site), allow_pickle=True)
            else:
                self.paths, self.text_labels = np.load('../data/DomainNet/{}_test.pkl'.format(site), allow_pickle=True)
            
        label_dict = {'bird':0, 'feather':1, 'headphones':2, 'ice_cream':3, 'teapot':4, 'tiger':5, 'whale':6, 'windmill':7, 'wine_glass':8, 'zebra':9}     
        # label_dict = {'toilet':0, 'duck':1, 'binoculars':2, 'crayon':3, 'couch':4, 'stop_sign':5, 'wristwatch':6, 'lighter':7, 'pig':8, 'bulldozer':9, 'rollerskates':10, 'scissors':11, 'cannon':12, 'leg':13, 'face':14, 'beach':15, 'skateboard':16, 't-shirt':17, 'clock':18, 'light_bulb':19, 'dresser':20, 'purse':21, 'coffee_cup':22, 'dishwasher':23, 'bicycle':24, 'map':25, 'picture_frame':26, 'hamburger':27, 'candle':28, 'hat':29, 'bracelet':30, 'monkey':31, 'moustache':32, 'zebra':33, 'crown':34, 'ambulance':35, 'windmill':36, 'nose':37, 'spider':38, 'fire_hydrant':39, 'strawberry':40, 'nail':41, 'telephone':42, 'submarine':43, 'blackberry':44, 'owl':45, 'rake':46, 'tooth':47, 'apple':48, 'octagon':49, 'camel':50, 'scorpion':51, 'megaphone':52, 'tent':53, 'cruise_ship':54, 'postcard':55, 'piano':56, 'remote_control':57, 'sailboat':58, 'bed':59, 'bucket':60, 'dragon':61, 'The_Mona_Lisa':62, 'bathtub':63, 'fireplace':64, 'watermelon':65, 'garden':66, 'flower':67, 'cello':68, 'garden_hose':69, 'donut':70, 'snake':71, 'house_plant':72, 'stitches':73, 'lipstick':74, 'mailbox':75, 'umbrella':76, 'chair':77, 'hedgehog':78, 'crocodile':79, 'string_bean':80, 'suitcase':81, 'skyscraper':82, 'belt':83, 'tractor':84, 'frying_pan':85, 'feather':86, 'baseball':87, 'pear':88, 'axe':89, 'lantern':90, 'wine_glass':91, 'table':92, 'fork':93, 'kangaroo':94, 'oven':95, 'rhinoceros':96, 'snail':97, 'pizza':98, 'elbow':99, 'eyeglasses':100, 'fish':101, 'hammer':102, 'knife':103, 'diving_board':104, 'sleeping_bag':105, 'drums':106, 'helmet':107, 'mermaid':108, 'shorts':109, 'marker':110, 'carrot':111, 'pliers':112, 'steak':113, 'washing_machine':114, 'saw':115, 'pineapple':116, 'horse':117, 'octopus':118, 'pond':119, 'syringe':120, 'floor_lamp':121, 'spoon':122, 'birthday_cake':123, 'ceiling_fan':124, 'pillow':125, 'jacket':126, 'shovel':127, 'bus':128, 'hourglass':129, 'mushroom':130, 'pickup_truck':131, 'river':132, 'pool':133, 'eye':134, 'rainbow':135, 'hot_air_balloon':136, 'pencil':137, 'swing_set':138, 'canoe':139, 'diamond':140, 'triangle':141, 'roller_coaster':142, 'potato':143, 'peanut':144, 'hockey_stick':145, 'castle':146, 'squiggle':147, 'blueberry':148, 'ant':149, 'dolphin':150, 'anvil':151, 'fence':152, 'onion':153, 'rabbit':154, 'keyboard':155, 'flip_flops':156, 'rain':157, 'animal_migration':158, 'book':159, 'bowtie':160, 'paintbrush':161, 'lion':162, 'bench':163, 'cooler':164, 'mountain':165, 'shoe':166, 'television':167, 'mug':168, 'peas':169, 'trombone':170, 'trumpet':171, 'microphone':172, 'yoga':173, 'church':174, 'bird':175, 'microwave':176, 'laptop':177, 'speedboat':178, 'smiley_face':179, 'bandage':180, 'panda':181, 'bottlecap':182, 'hockey_puck':183, 'matches':184, 'traffic_light':185, 'broccoli':186, 'asparagus':187, 'see_saw':188, 'lollipop':189, 'raccoon':190, 'van':191, 'sweater':192, 'goatee':193, 'snowman':194, 'cookie':195, 'parrot':196, 'bee':197, 'toe':198, 'cup':199, 'crab':200, 'palm_tree':201, 'clarinet':202, 'lobster':203, 'angel':204, 'The_Eiffel_Tower':205, 'police_car':206, 'teapot':207, 'snowflake':208, 'cow':209, 'toothbrush':210, 'squirrel':211, 'basket':212, 'cactus':213, 'circle':214, 'hot_tub':215, 'sun':216, 'bread':217, 'camera':218, 'brain':219, 'moon':220, 'dog':221, 'popsicle':222, 'airplane':223, 'train':224, 'guitar':225, 'hospital':226, 'soccer_ball':227, 'sock':228, 'arm':229, 'grapes':230, 'tennis_racquet':231, 'mosquito':232, 'compass':233, 'stairs':234, 'mouse':235, 'sheep':236, 'boomerang':237, 'ear':238, 'firetruck':239, 'giraffe':240, 'sink':241, 'drill':242, 'alarm_clock':243, 'eraser':244, 'The_Great_Wall_of_China':245, 'fan':246, 'camouflage':247, 'envelope':248, 'toaster':249, 'finger':250, 'car':251, 'basketball':252, 'star':253, 'cloud':254, 'house':255, 'baseball_bat':256, 'knee':257, 'toothpaste':258, 'headphones':259, 'power_outlet':260, 'flashlight':261, 'whale':262, 'violin':263, 'flying_saucer':264, 'broom':265, 'snorkel':266, 'foot':267, 'paper_clip':268, 'chandelier':269, 'stethoscope':270, 'streetlight':271, 'necklace':272, 'tiger':273, 'barn':274, 'mouth':275, 'leaf':276, 'line':277, 'stereo':278, 'penguin':279, 'computer':280, 'key':281, 'saxophone':282, 'screwdriver':283, 'lightning':284, 'school_bus':285, 'sea_turtle':286, 'bridge':287, 'motorbike':288, 'skull':289, 'teddy-bear':290, 'pants':291, 'hot_dog':292, 'wheel':293, 'ladder':294, 'elephant':295, 'grass':296, 'aircraft_carrier':297, 'underwear':298, 'waterslide':299, 'wine_bottle':300, 'dumbbell':301, 'frog':302, 'flamingo':303, 'zigzag':304, 'bat':305, 'sword':306, 'passport':307, 'hurricane':308, 'tree':309, 'truck':310, 'paint_can':311, 'cake':312, 'banana':313, 'ice_cream':314, 'helicopter':315, 'hexagon':316, 'swan':317, 'hand':318, 'backpack':319, 'parachute':320, 'jail':321, 'sandwich':322, 'campfire':323, 'shark':324, 'golf_club':325, 'vase':326, 'rifle':327, 'tornado':328, 'square':329, 'lighthouse':330, 'calendar':331, 'butterfly':332, 'ocean':333, 'cell_phone':334, 'bush':335, 'beard':336, 'bear':337, 'harp':338, 'stove':339, 'spreadsheet':340, 'radio':341, 'door':342, 'calculator':343, 'cat':344}
        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)
        
        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_transform():
    transform_train = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.ToTensor(),
    ])

    return transform_train, transform_test


def prepare_data_fed(args, train=True, debug=False):
    data_base_path = '../data'
    transform_train, transform_test = get_transform()

    train_loaders = []
    val_loaders = []
    test_loaders = []
    for ds in args.datasets:
        if train:
            trainset = DomainNetDataset(data_base_path, ds, transform=transform_train)
            rng = np.random.default_rng(2021)
            idx = rng.choice(len(trainset), size=len(trainset), replace=False)
            if debug:
                idx = idx[:50]
                print(idx)
            val_len = int(0.2*len(idx))

            valset = torch.utils.data.Subset(trainset, idx[-val_len:])
            trainset   = torch.utils.data.Subset(trainset, idx[:-val_len])
            
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
            val_loader   = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False)

            train_loaders.append(train_loader)
            val_loaders.append(val_loader)

        else:
            testset = DomainNetDataset(data_base_path, ds, transform=transform_test, train=False)
            test_loader  = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
            test_loaders.append(test_loader)

    if train:
        return train_loaders, val_loaders
    else:
        return test_loaders


def prepare_data(args, train=True, debug=False):
    data_base_path = '../data'
    transform_train, transform_test = get_transform()

    if train:
        trainset = DomainNetDataset(data_base_path, site='centralized', transform=transform_train)
        rng = np.random.default_rng(2021)
        idx = rng.choice(len(trainset), size=len(trainset), replace=False)   
        if debug:
            idx = idx[:100]
            print(idx)
        val_len = int(0.2*len(idx))


        valset = torch.utils.data.Subset(trainset, idx[-val_len:])
        trainset   = torch.utils.data.Subset(trainset, idx[:-val_len])

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
        val_loader   = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False)

        return train_loader, val_loader

    else:
        testset = DomainNetDataset(data_base_path, site='centralized', transform=transform_test, train=False)
        test_loader  = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        return test_loader
    

    return train_loader, val_loader


def plot_fed(args, arr, mode):
    plt.figure(figsize=(20, 10))
    plt.subplot(211)
    y = np.mean(arr['train'], axis=0)
    plt.plot(range(1, arr['train'].shape[1]+1), y, label='mean')
    for idx, ds in enumerate(args.datasets):
        plt.plot(range(1, arr['train'].shape[1]+1), arr['train'][idx, :],  label=ds)
    plt.xlabel("epoch")
    plt.ylabel(mode)
    plt.legend()

    plt.subplot(212)
    y = np.mean(arr['val'], axis=0)
    plt.plot(range(1, arr['val'].shape[1]+1), y, label='mean')
    for ds in args.datasets:
        plt.plot(range(1, arr['val'].shape[1]+1), arr['val'][idx, :], label=ds)
    plt.xlabel("epoch")
    plt.ylabel(mode)
    plt.legend()

    plt.savefig(args.fig_path + "_" + mode + ".png")
    # plt.show()

def plot(args, arr, mode):

    plt.figure(figsize=(20, 10))
    plt.subplot(211)
    plt.plot(range(1, arr['train'].shape[0]+1), arr['train'], label='mean')
    plt.xlabel("epoch")
    plt.ylabel(mode)
    plt.legend()

    plt.subplot(212)
    plt.plot(range(1, arr['val'].shape[0]+1), arr['val'], label='mean')
    plt.xlabel("epoch")
    plt.ylabel(mode)
    plt.legend()

    plt.savefig(args.fig_path + "_" + mode + ".png")


