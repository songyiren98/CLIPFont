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
    

#     prompts = ["English word, tree","English word, Lightning shape", "English word, water droplets","English word, Skeleton, scary, evil","English word, straight metal frame","English word, green snakes","English word, Ghost","English word, Blue Swirl","English word, Steampunk","English word, Cthulhu","English word, fire font, red and yellow fire","English word, red leave"]
#     prompts = ["English word, Golden lighting","English word, metal gear","English word, Cyberpunk","English word, Van Gogh's starry sky","English word, Straight wood strips, wood grain","English word, City skyline, buildings","English word, Palm leaves"]
#     prompts = ["English word, Christmas","English word, Chinese new year","English word, Knife and Sword","English word, Violets"]
#     prompts = ["Chinese character, Steampunk"]
#     prompts = ["English word, Christmas","English word, Chinese new year","English word, Knife and Sword","English word, Violets"]
    
#     prompts = ["Chinese character, Steampunk","Chinese character, tree","Chinese character, Lightning shape", "Chinese character, water droplets","Chinese character, Skeleton, scary, evil","Chinese character, straight metal frame","Chinese character, green snakes","Chinese character, Ghost","Chinese character, Blue Swirl","Chinese character, Steampunk","Chinese character, Cthulhu","Chinese character, fire font, red and yellow fire","Chinese character, red leave","Chinese character, Golden lighting","Chinese character, metal gear","Chinese character, Cyberpunk","Chinese character, Van Gogh's starry sky","Chinese character, Straight wood strips, wood grain","Chinese character, City skyline, buildings","Chinese character, Palm leaves","Chinese character, Christmas","Chinese character, Chinese new year","Chinese character, Knife and Sword","Chinese character, Violets"]


    # prompts = ["English word, Lightning shape", "English word, water droplets","English word, Steampunk","English word, Cthulhu","English word, fire font"]
    
    # prompts = ["Chinese character, Steampunk","Cthulhu, Chinese character","Flower, Chinese character","Chinese character, tree","Chinese character, City skyline, buildings","Chinese character, Palm leaves","Chinese character, Violets"]
    # prompts = ["Cthulhu, tree"]
    
    # prompts = ["English word, Lightning shape", "English word, water droplets","English word, Steampunk","English word, Cthulhu","English word, fire font","English word, Palm leaves"]
    # 1
    # prompts = ["English word, fire font"]
    
    prompts = ["Cyberpunk"]
    # prompts = ["Cthulhu, Chinese character"]
    
    # prompt2 = "English word, Skeleton, scary, evil"
    # prompt3 = "black robe"
    # prompt4 = "messy"
    
    source = "A photo"
    for prompt in prompts:

        with torch.no_grad():
            template_text = compose_text_with_templates(prompt, imagenet_templates)
            tokens = clip.tokenize(template_text).to(pydiffvg.get_device())
            text_features = clip_model.encode_text(tokens).detach()
            text_features = text_features.mean(axis=0, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            print(text_features)

            # template_text1 = compose_text_with_templates(prompt2, imagenet_templates)
            # tokens1 = clip.tokenize(template_text1).to(pydiffvg.get_device())
            # text_features1 = clip_model.encode_text(tokens1).detach()
            # text_features1 = text_features1.mean(axis=0, keepdim=True)
            # text_features1 /= text_features1.norm(dim=-1, keepdim=True)

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
        pydiffvg.imwrite(img.cpu(), 'results/refine_svg/init.png', gamma=gamma)

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



        # Optimize piont:0.1 color: 0.01
        points_optim = torch.optim.Adam(points_vars, lr= 0.1)
        color_optim = torch.optim.Adam(color_vars, lr=0.01)

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
            pydiffvg.imwrite(img.cpu(), 'results/refine_svg/iter_{}.png'.format(t), gamma=gamma)
            if t == 70:
                pydiffvg.imwrite(img.cpu(), 'results/body/iter_70{}.png'.format(t), gamma=gamma)
            if t == 140:
                pydiffvg.imwrite(img.cpu(), 'results/body/iter_140{}.png'.format(t), gamma=gamma)
            img = img[:, :, :3]
            # Convert img from HWC to NCHW
            img = img.unsqueeze(0)
            img = img.permute(0, 3, 1, 2) # NHWC -> NCHW

            if args.use_lpips_loss:
                loss =  1000*(img - target).pow(2).mean()
            # else:
            #     loss = 1*(img - target).pow(2).mean()
#             print('Similar loss:', loss.item())

            # new Style loss ******************************************
            target = img
            target.requires_grad_(True)

            loss_patch=0 
            img_proc =[]
            
            for n in range(num_crops):
                target_crop = cropper(target)
                target_crop = augment(target_crop)
                img_proc.append(target_crop)
            
#             target_head = target[:,:,100:400,50:350]
#             # crop head imgs
#             for n in range(num_crops):
#                 target_crop = cropper3(target_head)
#                 target_crop = augment(target_crop)
#                 img_proc.append(target_crop)
                
            
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

            text_direction = (1*text_features - text_source).repeat(image_features.size(0),1)

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

            text_direction2 = (1*text_features - text_source).repeat(glob_features.size(0),1)

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

            if  t == args.num_iter - 1:
                pydiffvg.save_svg('results/body/{}350.svg'.format(prompt),
                                  canvas_width, canvas_height, shapes, shape_groups)
            if t == 170:
                pydiffvg.save_svg('results/body/{}270.svg'.format(prompt),
                                  canvas_width, canvas_height, shapes, shape_groups)
            if t == 170:
                pydiffvg.save_svg('results/body/{}170.svg'.format(prompt),
                                  canvas_width, canvas_height, shapes, shape_groups)
            if t == 130:
                pydiffvg.save_svg('results/body/{}130.svg'.format(prompt),
                                  canvas_width, canvas_height, shapes, shape_groups)
            if t == 100:
                pydiffvg.save_svg('results/body/{}100.svg'.format(prompt),
                                  canvas_width, canvas_height, shapes, shape_groups)
            if t == 80:
                pydiffvg.save_svg('results/body/{}70.svg'.format(prompt),
                                  canvas_width, canvas_height, shapes, shape_groups)
            if t == 70:
                pydiffvg.save_svg('results/body/{}60.svg'.format(prompt),
                                  canvas_width, canvas_height, shapes, shape_groups)
            if t == 60:
                pydiffvg.save_svg('results/body/{}50.svg'.format(prompt),
                                  canvas_width, canvas_height, shapes, shape_groups)
            if t == 40:
                pydiffvg.save_svg('results/body/{}40.svg'.format(prompt),
                                  canvas_width, canvas_height, shapes, shape_groups)
            if t == 30:
                pydiffvg.save_svg('results/body/{}30.svg'.format(prompt),
                                  canvas_width, canvas_height, shapes, shape_groups)
            if t == 20:
                pydiffvg.save_svg('results/body/{}30.svg'.format(prompt),
                                  canvas_width, canvas_height, shapes, shape_groups)     
            if t == 2:
                pydiffvg.save_svg('results/body/{}2.svg'.format(prompt),
                                  canvas_width, canvas_height, shapes, shape_groups)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("svg", help="source SVG path")
    parser.add_argument("target", help="target image path", default="imgs/ABCD1.svg")
    parser.add_argument("--use_lpips_loss", dest='use_lpips_loss', action='store_true')
    parser.add_argument("--num_iter", type=int, default=150)
    
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
    parser.add_argument('--lambda_patch', type=float, default=150,
                        help='PatchCLIP loss parameter')
    parser.add_argument('--lambda_dir', type=float, default=30,
                        help='directional loss parameter')
#     parser.add_argument('--lambda_c', type=float, default=1.5,
#                         help='content loss parameter')
    parser.add_argument('--crop_size', type=int, default=230,
                        help='cropped image size')
    parser.add_argument('--num_crops', type=int, default=128,
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
