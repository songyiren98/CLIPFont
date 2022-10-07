import pydiffvg
import torch
import skimage
import skimage.io
import random
import ttools.modules
import argparse
import math
from matplotlib import pyplot as plt
import torch.nn.functional as F
import os
import cv2
import dlib
import PIL 
from PIL import Image
import numpy as np
import torchvision

import clip
import torch.nn
import torch.optim as optim
from torchvision import transforms, models

import utils
from template import imagenet_templates
from torchvision import utils as vutils
from torchvision.transforms.functional import adjust_contrast
# from xing_loss import xing_loss
from diffvg_helper import decouple_shape_groups

gamma = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def img_denormalize(image):
    mean=torch.tensor([0.485, 0.456, 0.406]).to(pydiffvg.get_device())
    std=torch.tensor([0.229, 0.224, 0.225]).to(pydiffvg.get_device())
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)
    image = image*std + mean
    return image

def img_normalize(image):
    mean=torch.tensor([0.485, 0.456, 0.406]).to(pydiffvg.get_device())
    std=torch.tensor([0.229, 0.224, 0.225]).to(pydiffvg.get_device())
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)
    image = (image-mean)/std
    return image

def clip_normalize(image,device):
    image = F.interpolate(image,size=224,mode='bicubic')
    mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(pydiffvg.get_device())
    std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(pydiffvg.get_device())
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)
    image = (image-mean)/std
    return image
    
def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)  
    return loss_var_l2

def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]
    
def findcolor(prompt):
    pic = np.ones((1,224,224,3))
    
def readimg(name):
    path = "results/painterly_rendering/iter_" + name
    content_image = torch.from_numpy(skimage.io.imread(path)).to(torch.float32) / 255.0
    content_image = content_image.pow(gamma)
    content_image = content_image.to(pydiffvg.get_device())
    content_image = content_image.unsqueeze(0)
    content_image = content_image.permute(0, 3, 1, 2) # NHWC -> NCHW
    return content_image

def main(args):
    
    num_crops = args.num_crops
    exp_name = "exp1"
    lr = 5e-4
    thresh = 0.0
    content_weight = 1.5
    style_weight = 1

    basename = os.path.splitext(os.path.basename(args.svg))[0]
    output_dir_base = os.path.join(os.path.dirname(__file__), '..', 'results', 'edit_svg', basename)
    
    # back = cv2.imread('imgs/face2.jpeg')
    # back = back[:,:,::-1]
#     back = back.transpose(2, 0, 1)
    # back = back/255
    # back = torch.from_numpy(back).float().to(pydiffvg.get_device())
    
    # new part *******************************

    content_path = args.target
    content_image = utils.load_image2(content_path, img_size=512)
    content = args.content_name
    exp = args.exp_name

    content_image = content_image.to(pydiffvg.get_device())

#     content_features = utils.get_features(img_normalize(content_image), VGG)
    output_image = content_image

    # data augment function
    cropper = transforms.Compose([
        transforms.RandomCrop(args.crop_size)
    ])
    cropper2 = transforms.Compose([
        transforms.RandomCrop(500,padding= 100, fill=255,padding_mode='constant')
    ])
    cropper3 = transforms.Compose([
        transforms.RandomCrop(160,padding= 35, fill=255,padding_mode='constant'),
        transforms.RandomHorizontalFlip(p=0.3)
    ])
    augment = transforms.Compose([
        transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.3),
        transforms.Resize(512)
    ])

    clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)
    
#     prompt = "Neanderthal"  
#     prompt = "EVA second machine color matching, purple and green"
#     prompt = "Hellboy"
#     prompt = "A building designed by Mies van der Rohe"
#     prompt = "Wolverine"
#     prompt = "Leonardo DiCaprio"
#     prompt = "Will Smith"
#     prompt = "Jocker, Heath Ledger"
#     prompt = "Jocker, Joaquin Phoenix"
#     prompt = "Voldemort"
#     prompt = "Hellboy"
    prompt = "Pop style"
#     prompts = ["Thanos","Obama","Morgan Freeman","Frankenstein","Einstein","Hitler","Tom Cruz","Donald Trump","Bruce Lee","Benedict Cumberbatch","Schwarzenegger, The Terminator","Johnny Depp, Captain Jack","Rowan Atkinson, Mr Bean","Anthony Hopkins, Hannibal","Harry Potter","Chris Evans","Jackie Chan"]
#     prompts = ["White walker","Will Smith","Edogawa Conan","Avatar"]
#     prompts = ["Endgame, Thanos","Rocky,Tom Hiddleston","Thor, the Avengers"]
#     prompts = ["Tolkien elf","Old-timey photograph"]
#     prompts = ["Optimus Prime","Bumblebee, transformer","Alien"]
#     prompts = ["Gorilla","Dog face","Cat face","Pig face","Lion face","Owl face","bronze sculpture"," White plaster statue"]
#     prompts = ["Skeleton"]
#     prompts = ["Jocker, Heath Ledger"]
#     prompts = ["Superman","Batman","Ironman","Thanos","Obama","Morgan Freeman","Frankenstein","Einstein","Hitler","Tom Cruz","Donald Trump","Bruce Lee","Benedict Cumberbatch","Schwarzenegger, The Terminator","Johnny Depp, Captain Jack","Rowan Atkinson, Mr Bean","Anthony Hopkins, Hannibal","Harry Potter","Chris Evans","Jackie Chan","Endgame, Thanos","Rocky,Tom Hiddleston","Thor, the Avengers","Optimus Prime","Bumblebee, transformer","Alien","Tolkien elf"]
#     prompts = ["Purple face, Endgame, Thanos"]
#     prompts = ["Gorilla","Dog face","Cat face","Pig face","Lion face","Owl face","bronze sculpture"," White plaster statue"]
#     prompts = ["Jackie Chan","Schwarzenegger, The Terminator"]
#     prompts = ["Optimus Prime","Endgame, Thanos","Starlord","Spiderman","Captain America","Venom, spiderman","Flash, DC","Green Goblin","Cyborg","Aquaman","Green Lantern","Daredevil"," Skeleton","Red Skull","werewolf","Hulk","War machine, ironman","Army, soldier","Astronaut, NASA","The Lizard, Mavel","Sandman","Fireman","Avatar","King Kong","Dwayne Johnson","scarecrow","Mummy","SEALs","Buzz Lightyear","Sheriff woody","Tiga Ultraman","Ultraman","Sun Wukong","Monkey King","Samurai","Takeda Nobunaga","lawyer in suit","Man with tattoo","US Air Force pilot","Zombi","Doctor strange","Neanderthal","Groot","Will Smith","White Walker","Voldemort","Frankenstein","Leonardo DiCaprio","Wolverine","Hellboy","Superman","Batman","Ironman","Thanos","Johnny Depp, Captain Jack","Bumblebee, transformer","Alien","Tolkien elf","Rocky,Tom Hiddleston","Thor, the Avengers","Rowan Atkinson, Mr Bean","Donald Trump","Jackie Chan","Schwarzenegger, The Terminator","Skull","Bruce Lee"]
#     prompts = ["Optimus Prime","Endgame, Thanos","War machine, ironman","Sun Wukong","Superman","Hellboy","US Air Force pilot","lawyer in suit","Schwarzenegger, The Terminator","Batman","Groot","Astronaut, NASA","Mummy","Neanderthal","Spiderman"]
#     prompts = ["Voldemort","Samurai","Takeda Nobunaga","Leonardo DiCaprio in suit","Ironman","White Walker","Flash, DC","Buzz Lightyear","Sheriff woody","Tiga Ultraman"]
#     prompts = ["Doctor Strange, Benedict Cumberbatch","Tom Holland in Spiderman suit"]
#     prompts = ["Matthew McConaughey in space suit, Interstellar","Chris Evans in Captain America uniform","Tom Cruise in US Air Force pilot, In movie Top Gun: Maverick","The Joker played by Heath Ledger"]
#     prompts = ["The Joker played by Heath Ledger, Purple suit jacket","star-lord in leather"," Henry Cavill in Superman suit"]
#     prompts = ["The Joker played by Heath Ledger,Purple suit jacket"]
#     prompts = ["Black Widow played by Scarlett Johansson, in black suit","Audrey Hepburn","Scarlet Witch played by Elizabeth Olsen, in red leather jacket"]
#     prompts = ["Harley Quinn","Catwoman played by Anne Hathaway","Agent Carter","Captain marvel","Cleopatra","Wonder Woman played by Gal Gadot"]
#     prompts = ["Sailor suit","スク水","Supergirl","Nami in One Piece","Bunny","Nurse","Teacher","Maid outfit"]
#     prompts = ["PrincessAeolian, Wlop","Machina","Spider Woman, Felicia Hardy","Poison Ivy","Mystique","Invisible Woman, Jessica Alba","She-hulk"]
#     prompts = ["Invisible Woman, Jessica Alba in Fantastic Four uniforms","Bunny girl, black stockings","Gamora, Mavel","Nebula, Mavel","Valkyrie in thor movies","Hela the god of death","Medusa, the Gorgon", "Hermione Grant in Hogwarts uniform"]
#     prompts = ["Elektra, DC superhero","Harley Quinn played by Margot Robbie","Chinese Peking Opera Actors","Rika Mori in a kimono"]
#     prompts = ["Future policewoman in police uniform and sunglasses"]
#     prompts = ["Cat face","Lion face","Tiger face","Eagle face"]
#     prompts = ["Optimus Prime","Endgame, Thanos","Starlord","Spiderman","Captain America","Venom, spiderman","Flash, DC","Green Goblin","Cyborg","Aquaman","Green Lantern","Daredevil"," Skeleton","Red Skull","werewolf","Hulk","War machine, ironman","Army, soldier","Astronaut, NASA","The Lizard, Mavel","Sandman","Fireman","Avatar","King Kong","Dwayne Johnson","scarecrow","Mummy","SEALs","Buzz Lightyear","Sheriff woody","Tiga Ultraman","Ultraman","Sun Wukong","Monkey King","Samurai","Takeda Nobunaga","lawyer in suit","Man with tattoo","US Air Force pilot","Zombi","Doctor strange","Neanderthal","Groot","Will Smith","White Walker","Voldemort","Frankenstein","Leonardo DiCaprio","Wolverine","Hellboy","Superman","Batman","Ironman","Thanos","Johnny Depp, Captain Jack","Bumblebee, transformer","Alien","Tolkien elf","Rocky,Tom Hiddleston","Thor, the Avengers","Rowan Atkinson, Mr Bean","Donald Trump","Jackie Chan","Schwarzenegger, The Terminator","Skull","Bruce Lee"]
#     prompts = ["Angry","Sad","Arrogant","Fear","surprised","disgusted","Sleepy","teary-eyed","Cute","Young","Aged"]
#     prompts = ["Red hair","Make up","blue eyes"]
#     prompts = ["The Joker played by Heath Ledger, Purple suit jacket","star-lord in leather","Henry Cavill in Superman suit","Superman played by Henry Cavill"]
#     prompts = ["Daredevil","Groot","White Walker","Voldemort in black robe","Sun Wukong","Batman","Hulk","Doctor Strange, Benedict Cumberbatch","Matthew McConaughey in space suit, Interstellar","Tom Cruise in US Air Force pilot, In movie Top Gun: Maverick"]
#     prompts = ["Doctor strange, Benedict Cumberbatch","Doctor strange","Jocker, Heath ledger","Groot","Voldemort","Zombie"]
#     prompts = ["Optimus Prime","Endgame, Thanos","Starlord","Spiderman","Captain America","Venom, spiderman","Flash, DC","Green Goblin","Cyborg","Aquaman","Green Lantern","Daredevil"," Skeleton","Red Skull","werewolf","Hulk","War machine, ironman","Army, soldier","Astronaut, NASA","The Lizard, Mavel","Sandman","Fireman","Avatar","King Kong","Dwayne Johnson","scarecrow","Mummy","SEALs","Buzz Lightyear","Sheriff woody","Tiga Ultraman","Ultraman","Sun Wukong","Monkey King","Samurai","Takeda Nobunaga","lawyer in suit","Man with tattoo","US Air Force pilot","Zombi","Doctor strange","Neanderthal","Groot","Will Smith","White Walker","Voldemort","Frankenstein","Leonardo DiCaprio","Wolverine","Hellboy","Superman","Batman","Ironman","Thanos","Johnny Depp, Captain Jack","Bumblebee, transformer","Alien","Tolkien elf","Rocky,Tom Hiddleston","Thor, the Avengers","Rowan Atkinson, Mr Bean","Donald Trump","Jackie Chan","Schwarzenegger, The Terminator","Skull","Bruce Lee"]
#     prompts = ["English word, tree","English word, Lightning shape", "English word, water droplets","English word, Skeleton, scary, evil","English word, straight metal frame","English word, green snakes","English word, Ghost","English word, Blue Swirl","English word, Steampunk","English word, Cthulhu","English word, fire font, red and yellow fire","English word, red leave"]
#     prompts = ["English word, Golden lighting","English word, metal gear","English word, Cyberpunk","English word, Van Gogh's starry sky","English word, Straight wood strips, wood grain","English word, City skyline, buildings","English word, Palm leaves"]
#     prompts = ["English word, Christmas","English word, Chinese new year","English word, Knife and Sword","English word, Violets"]
#     prompts = ["Chinese character, Steampunk"]
#     prompts = ["English word, Christmas","English word, Chinese new year","English word, Knife and Sword","English word, Violets"]
    
#     prompts = ["Chinese character, Steampunk","Chinese character, tree","Chinese character, Lightning shape", "Chinese character, water droplets","Chinese character, Skeleton, scary, evil","Chinese character, straight metal frame","Chinese character, green snakes","Chinese character, Ghost","Chinese character, Blue Swirl","Chinese character, Steampunk","Chinese character, Cthulhu","Chinese character, fire font, red and yellow fire","Chinese character, red leave","Chinese character, Golden lighting","Chinese character, metal gear","Chinese character, Cyberpunk","Chinese character, Van Gogh's starry sky","Chinese character, Straight wood strips, wood grain","Chinese character, City skyline, buildings","Chinese character, Palm leaves","Chinese character, Christmas","Chinese character, Chinese new year","Chinese character, Knife and Sword","Chinese character, Violets"]
#     prompts =["Hot Fire weapons, Unreal engine","Forest attribute weapons, wooden handle, Unreal engine","Golden weapons, Unreal engine"]
#     prompts = ["Angry Peppa","Sad Peppa","Suprice Peppa","Scared Peppa","Happy Peppa"]
#     prompts = ["Will Smith","Obama","Elon Musk","Einstein"]
#     prompts = ["Doctor strange, Artstation","Doctor strange, Pinterest","Doctor strange, Unreal Engine"]
#     prompts = ["Scarlett Johansson","Tom Holland","Will Smith","Obama","Elon Musk","Einstein","Doctor strange, Artstation","Neanderthal","Groot","Beethoven"]
#     prompts = ["Newton","Nikola Tesla","Ludwig van Beethoven","Che Guevara","Skull"]
#     prompts = ["Karl Marx","Claude Elwood Shannon","Alan Mathison Turing"]
#     prompts = ["Doctor strange, Artstation","Tom Holland","Jocker, Heath Ledger","Groot","Nikola Tesla","Newton","Claude Elwood Shannon"]
#     prompts = ["Karl Marx","Alan Mathison Turing","Superman","Batman","Ironman","Endgame, Thanos","Starlord","Leonardo DiCaprio","Elon Musk","Einstein","Rowan Atkinson, Mr Bean","Bruce Lee","Neanderthal","Karl Marx","Franklin Delano Roosevelt","Churchill","Joseph Vissarionovich Stalin","Abraham Lincoln","Queen","Marie Curie","Che Guevara","Scarlett Johansson"]
#     prompts = ["Scarlett Johansson","Anne Hathaway","Elizabeth Olsen","Audrey Hepburn","Hermione Grant","Wonder woman"]
#     prompts = ["Cleopatra","Poison lvy","Goddess of Death, Hela","Machina","Zombie"]
#     prompts = ["Witch","Elf","Snow White","Little Red Riding Hood"]
#     prompts = ["Elf","Keanu Reeves, John Wick","John Wick","Aquaman","Sun Wukong","Neanderthal","Harry Potter","Voldemort","Dumbledore"]
#     prompts = ["Keanu Reeves"]
#     prompts = ["Hermione Granger","Scarlett Johansson","Anne Hathaway","Elizabeth Olsen","Audrey Hepburn","Wonder woman","Medusa, Gorgon","Jessica Alba","Greta Thunberg","Emma Watson","Eva Green","Ana de Armas","Chloe Bennet","Agent Carter","Harley Quinn","Disney animation, princess","Pixar Animation, Princess","Venus","Queen","Taylor Swift","Jennifer Connelly","Cleopatra","Goddess of Death, Hela","Machina","Zombie","Janpanese beauty"]
#     prompts = ["Alita: Battle Angel","Marilyn Monroe"]
#     prompts = ["Tom Holland","Will Smith","Obama","Elon Musk","Einstein","Doctor strange, Artstation","Optimus Prime","Endgame, Thanos","Starlord","Spiderman","Captain America","Venom, spiderman","Flash, DC","Green Goblin","Cyborg","Aquaman","Green Lantern","Daredevil"," Skeleton","Red Skull","werewolf","Hulk","War machine, ironman","Army, soldier","Astronaut, NASA","The Lizard, Mavel","Sandman","Fireman","Avatar","King Kong","Dwayne Johnson","scarecrow","Mummy","SEALs","Buzz Lightyear","Sheriff woody","Tiga Ultraman","Ultraman","Sun Wukong","Monkey King","Samurai","Takeda Nobunaga","lawyer in suit","Man with tattoo","US Air Force pilot","Zombi","Doctor strange","Neanderthal","Groot","Will Smith","White Walker","Voldemort","Frankenstein","Leonardo DiCaprio","Wolverine","Hellboy","Superman","Batman","Ironman","Thanos","Johnny Depp, Captain Jack","Bumblebee, transformer","Alien","Tolkien elf","Rocky,Tom Hiddleston","Thor, the Avengers","Rowan Atkinson, Mr Bean","Donald Trump","Jackie Chan","Schwarzenegger, The Terminator","Skull","Bruce Lee","Jocker, Heath Ledger"]
#     prompts = ["Jocker, Heath Ledger","Rocket Raccoon","The Winter Soldier","Tom Holland","Will Smith","Obama","Elon Musk","Einstein","Doctor strange, Artstation","Optimus Prime","Endgame, Thanos","Starlord","Captain America","Venom","Flash, DC","Astronaut, NASA","The Lizard, Mavel","Avatar","King Kong","Dwayne Johnson","scarecrow","Mummy","Buzz Lightyear","Sheriff woody","Tiga Ultraman","Ultraman","Sun Wukong","Samurai","Takeda Nobunaga","Zombi","Doctor strange","Neanderthal","Groot","White Walker","Voldemort","Frankenstein","Leonardo DiCaprio","Wolverine","Hellboy","Superman","Batman","Ironman","Johnny Depp, Captain Jack","Thor, the Avengers","Rowan Atkinson, Mr Bean","Donald Trump","Jackie Chan","Schwarzenegger, The Terminator","Skull","Bruce Lee","Kim Jong-un","LeBron James","David Beckham","Cristiano Ronaldo dos Santos Aveiro","Matthew McConaughey","Charlie Chaplin","Che Guevara","Robert Downey Jr","Stephen Curry","Yao Ming","Harry Potter","Shaquille O'Neal","Lionel Messi","Mark Elliot Zuckerberg","The Godfather, Marlon Brando","Hannibal","Pirate","Tom Cruise","Keanu Charles Reeves","Hugh Jackman","Morgan Freeman","Werewolf","vampire","Baron dracula","Spartan warrior","Sherlock Holmes","Dr. Watson","Future policeman","Pharaoh","Elf","Rick","Morty","Colonel Sanders, KFC","Burning skull", "Graduate with graduation cap","Rick","Morty","Colonel Sanders, KFC","Burning skull", "Graduate with graduation cap","Huck","Agent Patton","Aquaman","Bruce Lee","Doctor Who","Green devil","SpongeBob SquarePants","Squidward","Pigsy","The Sha Monk","F-22 Raptor"]
#     prompts = ["F-16 Fighting Falcon","Eurofighter Typhoon","Mikoyan MiG-29","camouflage painted aircraft"]
#     prompts = ["F-18 Super Hornet","F-15 Strike Eagle"]
#     prompts = ["Obama","Shaquille O'Neal","Dwyane Wade","Stephen Curry","LeBron James","Kobe Bryant","Dwayne Johnson","Will Smith","Morgan Freeman"]
#     prompts = ["Michael Jordan","Kevin Durant","Kyrie Irving","Ja Morant","Van Diesel","Jason Statham","Jason Statham","Michael Sylvester Gardenzio Stallone","Arnold Schwarzenegger","Kylian Mbappé","Usain Bolt","Yao Ming","Martin Luther King","Aamir Khan","Groot","Chow Yun Fat", "Tony Leung Chiu-wai", "Bruce Lee", "Stephen Chiau Sing Chi"]
    prompts = ["Jocker, Heath Ledger", "Jackie Chan"]
    prompts = args.prompts

#     prompt2 = "Green magic matrix on arm, red cloak"
#     prompt3 = "black robe"
#     prompt4 = "messy"
    
    source = "A photo"
    for prompt in prompts:
        output_dir = os.path.join(output_dir_base, prompt.replace(' ', '_').replace(',', ''))

        with torch.no_grad():
            template_text = compose_text_with_templates(prompt, imagenet_templates)
            tokens = clip.tokenize(template_text).to(pydiffvg.get_device())
            text_features = clip_model.encode_text(tokens).detach()
            text_features = text_features.mean(axis=0, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            print(text_features)

#             template_text1 = compose_text_with_templates(prompt2, imagenet_templates)
#             tokens1 = clip.tokenize(template_text1).to(pydiffvg.get_device())
#             text_features1 = clip_model.encode_text(tokens1).detach()
#             text_features1 = text_features1.mean(axis=0, keepdim=True)
#             text_features1 /= text_features1.norm(dim=-1, keepdim=True)

#             template_text2 = compose_text_with_templates(prompt3, imagenet_templates)
#             tokens2 = clip.tokenize(template_text2).to(pydiffvg.get_device())
#             text_features2 = clip_model.encode_text(tokens2).detach()
#             text_features2 = text_features2.mean(axis=0, keepdim=True)
#             text_features2 /= text_features2.norm(dim=-1, keepdim=True)

#             template_text3 = compose_text_with_templates(prompt4, imagenet_templates)
#             tokens3 = clip.tokenize(template_text3).to(pydiffvg.get_device())
#             text_features3 = clip_model.encode_text(tokens3).detach()
#             text_features3 = text_features3.mean(axis=0, keepdim=True)
#             text_features3 /= text_features3.norm(dim=-1, keepdim=True)

            template_source = compose_text_with_templates(source, imagenet_templates)
            tokens_source = clip.tokenize(template_source).to(pydiffvg.get_device())
            text_source = clip_model.encode_text(tokens_source).detach()
            text_source = text_source.mean(axis=0, keepdim=True)
            text_source /= text_source.norm(dim=-1, keepdim=True)
            source_features = clip_model.encode_image(clip_normalize(content_image,device))
            source_features /= (source_features.clone().norm(dim=-1, keepdim=True))   
        #new part end***************************************

#         perception_loss = ttools.modules.LPIPS().to(pydiffvg.get_device())

        target = torch.from_numpy(skimage.io.imread(args.target)).to(torch.float32) / 255.0
        target = target.pow(gamma)
        target = target.to(pydiffvg.get_device())
        target = target.unsqueeze(0)
        target = target.permute(0, 3, 1, 2) # NHWC -> NCHW

        canvas_width, canvas_height, shapes, shape_groups = \
            pydiffvg.svg_to_scene(args.svg)
        if args.decouple_color:
            shape_groups = decouple_shape_groups(shape_groups)
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        
#         ########INIT
# #         shapes = []
# #         shape_groups = []

#         for i in range(500):
#             num_segments = random.randint(3, 5)
#             num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
#             points = []
#             p0 = (random.random(), random.random())
#             points.append(p0)
#             for j in range(num_segments):
#                 radius = 0.1
#                 p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
#                 p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
#                 p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
#                 points.append(p1)
#                 points.append(p2)
#                 if j < num_segments - 1:
#                     points.append(p3)
#                     p0 = p3
#             points = torch.tensor(points)
#             points[:, 0] *= canvas_width
#             points[:, 1] *= canvas_height
#             path = pydiffvg.Path(num_control_points = num_control_points,
#                                  points = points,
#                                  stroke_width = torch.tensor(1.0),
#                                  is_closed = True)
#             shapes.append(path)
#             path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
#                                              fill_color = torch.tensor([random.random(),
#                                                                         random.random(),
#                                                                         random.random(),
#                                                                         random.random()/2]))
#             shape_groups.append(path_group)
#         ########INIT END

        render = pydiffvg.RenderFunction.apply
        img = render(canvas_width, # width
                     canvas_height, # height
                     2,   # num_samples_x
                     2,   # num_samples_y
                     0,   # seed
                     None, # bg
                     *scene_args)
        # The output image is in linear RGB space. Do Gamma correction before saving the image.
        pydiffvg.imwrite(img.cpu(), f'{output_dir}/init.png', gamma=gamma)

        points_vars = []
#         print(shapes)
        for path in shapes:
#             if tag == 'polygon':
            path.points.requires_grad = True
            points_vars.append(path.points)
        color_vars = {}
        # color lock
#         print(shape_groups)
        for group in shape_groups:
            group.fill_color.requires_grad = True
            color_vars[group.fill_color.data_ptr()] = group.fill_color
        color_vars = list(color_vars.values())



        # Optimize piont:1.0 color: 0.01
        points_optim = torch.optim.Adam(points_vars, lr=args.shape_lr)
        color_optim = torch.optim.Adam(color_vars, lr=args.color_lr)

#         points_optim = torch.optim.ASGD(points_vars, lr=100.0)
#         color_optim = torch.optim.ASGD(color_vars, lr=1)

        # Adam iterations.
        for t in range(args.num_iter):
            print('iteration:', t)
            points_optim.zero_grad()
            color_optim.zero_grad()
            # Forward pass: render the image.
            scene_args = pydiffvg.RenderFunction.serialize_scene(\
                canvas_width, canvas_height, shapes, shape_groups)
            img = render(canvas_width, # width
                         canvas_height, # height
                         2,   # num_samples_x
                         2,   # num_samples_y
                         0,   # seed
                         None, # bg
                         *scene_args)
            
            
            # Compose img with white background
            img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])*1

#             img = img[:, :, 3:4] * img[:, :, :3] + back *(1 - img[:, :, 3:4])
            
            # Save the intermediate render.
            img_save = img.detach().cpu()
            # pydiffvg.imwrite(img.cpu(), f'{output_dir}/iter_{str(t).zfill(4)}.png', gamma=gamma)
            img = img[:, :, :3]
            # Convert img from HWC to NCHW
            img = img.unsqueeze(0)
            img = img.permute(0, 3, 1, 2) # NHWC -> NCHW

            if args.use_lpips_loss:
                loss =  1000*(img - target).pow(2).mean()
            else:
                loss = 1*(img - target).pow(2).mean()
#             print('Similar loss:', loss.item())

            # new Style loss ******************************************
            target = img
            target.requires_grad_(True)

            loss_patch=0 
            img_proc =[]
            
#             for n in range(num_crops - 64):
#                 target_crop = cropper(target)
#                 target_crop = augment(target_crop)
#                 img_proc.append(target_crop)
            
            target_head = target[:,:,100:400,50:350]
            # crop head imgs
            for n in range(num_crops):
                target_crop = cropper3(target_head)
                target_crop = augment(target_crop)
                img_proc.append(target_crop)
                
            
#             for n in range(num_crops):
#                 target_crop = cropper(target)
#                 target_crop = augment(target_crop)
#                 img_proc.append(target_crop)

            img_proc = torch.cat(img_proc,dim=0)
            img_aug = img_proc

            image_features = clip_model.encode_image(clip_normalize(img_aug,device))
            image_features /= (image_features.clone().norm(dim=-1, keepdim=True))

            img_direction = (image_features-source_features)
            img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

        #         text_direction = (text_features - 1*text_source).repeat(image_features.size(0),1)

            text_direction = (text_features - text_source).repeat(image_features.size(0),1)

            text_direction /= text_direction.norm(dim=-1, keepdim=True)
            loss_temp = (1- torch.cosine_similarity(img_direction, text_direction, dim=1))
            loss_temp[loss_temp<args.thresh] =0
            loss_patch += loss_temp.mean()

            img_proc2 =[]
            for n in range(32):
                target_crop = cropper2(target)
                target_crop = augment(target_crop)
                img_proc2.append(target_crop)

            img_proc2 = torch.cat(img_proc2,dim=0)
            img_aug2 = img_proc2

            glob_features = clip_model.encode_image(clip_normalize(img_aug2,device))
            glob_features /= (glob_features.clone().norm(dim=-1, keepdim=True))

            text_direction2 = (text_features - text_source).repeat(glob_features.size(0),1)

            glob_direction = (glob_features-source_features)
            glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)


            loss_glob = (1- torch.cosine_similarity(glob_direction, text_direction2, dim=1)).mean()

            total_loss = args.lambda_patch*loss_patch + args.lambda_dir*loss_glob 
            
            print('clip loss:', total_loss.item())
            
            loss = total_loss 
#             loss = total_loss + xing_loss(points_vars) * 10000
        #         total_loss =  args.lambda_patch*loss_patch

    #         if t < 10:
    #             loss = loss*0.001
    #         if t < 20:
    #             loss = loss*0.01
    #         elif t < 30:
    #             loss = loss*0.02
    #         elif t < 40:
    #             loss = loss*0.03
    #         elif t < 50:
    #             loss = loss*0.04
    #         else:
    #             loss = loss*0.05
    #************************************
#             if args.use_lpips_loss:
#                 loss = 100*perception_loss(img, target2) + 5000*(img - target2).pow(2).mean()
#             else:
#                 loss = 1*(img - target2).pow(2).mean()
#             print('Similar loss:', loss.item())
            
#             if args.use_lpips_loss:
#                 lpipsloss = perception_loss(target,content_image)
#                 l2loss = (target-content_image).pow(2).mean()
#                 if t < 200:
#                     loss += lpipsloss*100  + l2loss*500000000
#                 elif t < 50:
#                     loss += lpipsloss*20  + l2loss*1500
#                 elif t < 70:
#                     loss += lpipsloss*10  + l2loss*1000
#                 else:
#                     loss += lpipsloss*5  + l2loss*800
                    
    #             loss = perception_loss(img, target)*0.05 + (img.mean() - target.mean()).pow(2) + (img-target).pow(2).mean()*0.5
#             else:
#             loss += (target-output_image).pow(2).mean()*3000

            print('total loss:', loss.item())

            #************************************


            # Backpropagate the gradients.
            loss.backward()

            # Take a gradient descent step.
            points_optim.step()
            color_optim.step()
            for group in shape_groups:
                group.fill_color.data.clamp_(0.0, 1.0)

            if t == args.num_iter - 1 or t % 10 == 0:
                pydiffvg.imwrite(img_save, f'{output_dir}/iter_{str(t).zfill(4)}.png', gamma=gamma)
                pydiffvg.save_svg(f'{output_dir}/{str(t).zfill(4)}.svg',
                                  canvas_width, canvas_height, shapes, shape_groups)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("svg", help="source SVG path")
    parser.add_argument("target", help="target image path", default="imgs/cartoonface.png")
    parser.add_argument("--use_lpips_loss", dest='use_lpips_loss', action='store_true')
    parser.add_argument("--num_iter", type=int, default=300)

    parser.add_argument("--prompts", type=str, nargs='+', required=True)
    parser.add_argument("--shape_lr", type=float, default=0.2)
    parser.add_argument("--color_lr", type=float, default=0.01)
    parser.add_argument("--decouple_color", action='store_true')
    

    parser.add_argument("--use_deeptransfer", dest='use_deeptransfer', action='store_true')
    parser.add_argument('--content_path', type=str, default="imgs/cartoonface.png",
                        help='Image resolution')
    parser.add_argument('--content_name', type=str, default="face",
                        help='Image resolution')
#*******************************

    parser.add_argument('--exp_name', type=str, default="exp1",
                        help='Image resolution')
    parser.add_argument('--text', type=str, default="Starry Night by Vincent van gogh",
                        help='Image resolution')
    parser.add_argument('--lambda_tv', type=float, default=2e-5,
                        help='total variation loss parameter')
    parser.add_argument('--lambda_patch', type=float, default=50,
                        help='PatchCLIP loss parameter')
    parser.add_argument('--lambda_dir', type=float, default=100,
                        help='directional loss parameter')
#     parser.add_argument('--lambda_c', type=float, default=1.5,
#                         help='content loss parameter')
    parser.add_argument('--crop_size', type=int, default=350,
                        help='cropped image size')
    parser.add_argument('--num_crops', type=int, default=64,
                        help='number of patches')
    parser.add_argument('--img_size', type=int, default=512,
                        help='size of images')
    parser.add_argument('--max_step', type=int, default=200,
                        help='Number of domains')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Number of domains')
    parser.add_argument('--thresh', type=float, default=0.0,
                        help='Number of domains')
#**********************************

    args = parser.parse_args()
    main(args)
