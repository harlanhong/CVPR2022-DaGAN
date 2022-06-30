import os
import random
import numpy as np
import csv
import cv2
import pdb
from collections import defaultdict
import sys
from tqdm import tqdm
from PIL import Image
import depth
import torch.nn as nn
from skimage.transform import resize
import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import math
import imageio
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread,imsave
import pandas as pd
from scipy.spatial import ConvexHull
def count_test_video(path):
    vis = {video[:video.find('#',8)] for video in
                                os.listdir(path)}
    print(vis)
    print(len(vis))

def create_same_id_test_set(path):
    vis = os.listdir(path)
    videos = np.random.choice(vis, replace=False, size=100)
    f = open('./data/vox_evaluation.csv','w',encoding='utf-8')
    source = []
    driving = []
    csv_writer = csv.writer(f)
    csv_writer.writerow(["source","driving","frame"])
    for i in range(2083):
        v = np.random.choice(videos, replace=False, size=1)
        imgs = os.listdir(os.path.join(path,v[0]))
        pair = np.random.choice(imgs, replace=False, size=2)
        src = os.path.join(path,v[0],pair[0])
        dst = os.path.join(path,v[0],pair[1])
        source.append(src)
        driving.append(dst)
    sources = np.array(source).reshape(-1,1)
    driving = np.array(driving).reshape(-1,1)
    content = np.concatenate((sources,driving),1)
    csv_writer.writerows(content)
    f.close()

def create_cross_id_test_set(path):
    vis = os.listdir(path)
    ids2video = defaultdict(list)
    num = len('id10283')
    for vi in vis:
        ids2video[vi[:num]].append(vi)
    ids = list(ids2video.keys())
    videos = np.random.choice(vis, replace=False, size=100)
    f = open('./data/vox_cross_id_evaluation.csv','w',encoding='utf-8')
    source = []
    driving = []
    csv_writer = csv.writer(f)
    csv_writer.writerow(["source","driving","frame"])
    for i in range(2083):
        id = np.random.choice(ids, replace=False, size=1)
        vis = np.random.choice(ids2video[id[0]], replace=False, size=1)
        imgs = os.listdir(os.path.join(path,vis[0]))
        img = np.random.choice(imgs, replace=False, size=1)
        src = os.path.join(path,vis[0],img[0])

        other_id = list(set(ids).difference(set(id)))
        id = np.random.choice(other_id, replace=False, size=1)
        vis = np.random.choice(ids2video[id[0]], replace=False, size=1)
        imgs = os.listdir(os.path.join(path,vis[0]))
        img = np.random.choice(imgs, replace=False, size=1)
        dst = os.path.join(path,vis[0],img[0])

        source.append(src)
        driving.append(dst)
    sources = np.array(source).reshape(-1,1)
    driving = np.array(driving).reshape(-1,1)
    content = np.concatenate((sources,driving),1)
    csv_writer.writerows(content)
    f.close()

def concate_compared_results(resust_path,cp_path):
    imgs = os.listdir(resust_path)
    for im in tqdm(imgs):
        ours = cv2.imread(os.path.join(resust_path,im))
        fomm = cv2.imread(os.path.join(cp_path,im))
        rst = np.concatenate((ours,fomm),0).astype(np.uint8)
        cv2.imwrite(os.path.join('FID/compare',im),rst)
def render(path):
    depth_encoder = depth.ResnetEncoder(18, False).cuda()
    depth_decoder = depth.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4)).cuda()
    loaded_dict_enc = torch.load('depth/models/weights_19/encoder.pth')
    loaded_dict_dec = torch.load('depth/models/weights_19/depth.pth')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()}
    depth_encoder.load_state_dict(filtered_dict_enc)
    depth_decoder.load_state_dict(loaded_dict_dec)
    depth_encoder.eval()
    depth_decoder.eval()
    cvimg = cv2.resize(cv2.imread(path),(256,256))
    img = Image.open(path).convert('RGB').resize((256,256))
    tensor_img = T.ToTensor()(img).unsqueeze(0).cuda()
    outputs = depth_decoder(depth_encoder(tensor_img))
    depth_source = outputs[("disp", 0)][0]
    depth_source = depth_source.permute(1,2,0).detach().cpu().numpy()
    heatmap = depth_source/np.max(depth_source)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img1 = heatmap*0.6+cvimg
    cv2.imwrite('{}.jpg'.format(path),superimposed_img1)

def depth_gray(path):
    depth_encoder = depth.ResnetEncoder(18, False).cuda()
    depth_decoder = depth.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4)).cuda()
    loaded_dict_enc = torch.load('depth/models/weights_19/encoder.pth')
    loaded_dict_dec = torch.load('depth/models/weights_19/depth.pth')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()}
    depth_encoder.load_state_dict(filtered_dict_enc)
    depth_decoder.load_state_dict(loaded_dict_dec)
    depth_encoder.eval()
    depth_decoder.eval()
    img = Image.open(path).convert('RGB').resize((256,256))
    tensor_img = T.ToTensor()(img).unsqueeze(0).cuda()
    outputs = depth_decoder(depth_encoder(tensor_img))
    depth_source = outputs[("disp", 0)][0]
    depth_source = depth_source.permute(1,2,0).detach().cpu().numpy()*depth_source.permute(1,2,0).detach().cpu().numpy()
    heatmap = 1-depth_source/np.max(depth_source)
    heatmap = np.uint8(255 * heatmap)
    cv2.imwrite('heatmap.jpg',heatmap)

def depth_rgb(path):
    depth_encoder = depth.ResnetEncoder(18, False).cuda()
    depth_decoder = depth.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4)).cuda()
    loaded_dict_enc = torch.load('depth/models/weights_19/encoder.pth')
    loaded_dict_dec = torch.load('depth/models/weights_19/depth.pth')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()}
    depth_encoder.load_state_dict(filtered_dict_enc)
    depth_decoder.load_state_dict(loaded_dict_dec)
    depth_encoder.eval()
    depth_decoder.eval()
    img = Image.open(path).convert('RGB').resize((256,256))
    tensor_img = T.ToTensor()(img).unsqueeze(0).cuda()
    outputs = depth_decoder(depth_encoder(tensor_img))
    disp = outputs[("disp", 0)]
    # Saving colormapped depth image
    disp_resized = torch.nn.functional.interpolate(disp, (256, 256), mode="bilinear", align_corners=False)
    disp_resized_np = disp_resized.squeeze().detach().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='rainbow')

    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    plt.axis('off')
    plt.imshow(colormapped_im)

    # plt.colorbar(mapper)
    plt.savefig(path+'.pdf')
    # plt.savefig(path+'.png')

    plt.clf()


def process_celeV(path):
    train_path = os.path.join(path,'train')
    test_path = os.path.join(path,'test')
    ids = os.listdir(path)
    f = open('./data/celeV_cross_id_evaluation.csv','w',encoding='utf-8')

    # sample 2000 image sets from each identity
    # if not os.path.exists(train_path):
    #     os.makedirs(train_path)
    # if not os.path.exists(test_path):
    #     os.makedirs(test_path)
    source = []
    driving = []
    csv_writer = csv.writer(f)
    csv_writer.writerow(["source","driving","frame"])
    for i in range(2083):
        src_id = np.random.choice(ids, replace=False, size=1)
        imgs = os.listdir(os.path.join(path,src_id[0],'Image'))
        src_imgs = np.random.choice(imgs, replace=False, size=1)
        src = os.path.join(path,src_id[0],'Image',src_imgs[0])

        res_ids = list(set(ids).difference(set(src_id)))

        dst_id = np.random.choice(res_ids, replace=False, size=1)
        imgs = os.listdir(os.path.join(path,dst_id[0],'Image'))
        dst_imgs = np.random.choice(imgs, replace=False, size=1)
        dst = os.path.join(path,dst_id[0],'Image',dst_imgs[0])
        source.append(src)
        driving.append(dst)
    sources = np.array(source).reshape(-1,1)
    driving = np.array(driving).reshape(-1,1)
    content = np.concatenate((sources,driving),1)
    csv_writer.writerows(content)
    f.close()


def compare():
    x2face = '/data/fhongac/workspace/gitrepo/X2Face/UnwrapMosaic/FID/celebv'
    fomm = '/data/fhongac/workspace/gitrepo/first-order-model/FID/celebv'
    osfv = '/data/fhongac/workspace/gitrepo/One-Shot_Free-View_Neural_Talking_Head_Synthesis/FID/celebv'
    dagan = '/data/fhongac/workspace/src/parallel-fom-rgbd/log/vox-adv-256rgbd_kp_num15_rgbd_attnv2/celebv/concate'
    
    imgs = os.listdir(x2face)
    for i in tqdm(range(len(imgs))):
        im = imgs[i]
        img_x2face = os.path.join(x2face,im)
        img_x2face = cv2.imread(img_x2face)

        img_fomm = os.path.join(fomm,im)
        img_fomm = cv2.imread(img_fomm)

        img_osfv = os.path.join(osfv,im)
        img_osfv = cv2.imread(img_osfv)

        img_dagan = os.path.join(dagan,im)
        img_dagan = cv2.imread(img_dagan)


        img = np.vstack((img_x2face, img_fomm,img_osfv,img_dagan))
        cv2.imwrite('FID/multiMethod/{}.jpg'.format(i),img)
def aus(path):
    import cv2
    frame = cv2.imread(path)
    from feat import Detector
    detector = Detector()  
    # image_prediction = detector.detect_image(path)
    out1 = detector.detect_image('FID/source/0.jpg')
    out1.plot_aus(12, muscles={'all': "heatmap"}, gaze = None)
    plt.savefig('a.jpg')
    out2 = detector.detect_image('FID/source/1.jpg')
    p1 = out1.facepose.values
    p2 = out2.facepose.values
    
    # landmarks = detector.detect_landmarks(frame, face)  
    # score = detector.detect_aus(frame,landmarks[0])
def evaluate_PRMSE_AUCON():
    from feat import Detector
    detector = Detector()
    # x2face = '/data/fhongac/workspace/gitrepo/X2Face/UnwrapMosaic/FID'
    # fomm = '/data/fhongac/workspace/gitrepo/first-order-model/FID'
    # osfv = '/data/fhongac/workspace/gitrepo/One-Shot_Free-View_Neural_Talking_Head_Synthesis/FID'
    # dagan = '/data/fhongac/workspace/src/parallel-fom-rgbd/FID'
    # test = dagan
    path = sys.argv[1]
    imgs = os.listdir(path+'/gt')
    PRMSE = 0
    AUCON = 0
    counter = 1e-9
    CSIM = 0
    csim_counter = 1e-9
    ##########################################CSIM##############################################################
    from facenet_pytorch import MTCNN, InceptionResnetV1

    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(image_size=256, margin=0)

    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    #####################################################################################################################
    for im in tqdm(imgs):
        try:
            gt = os.path.join(path,'gt',im)
            gen = os.path.join(path,'generate',im)
            # gt_aus = detector.detect_aus(gt)
            # generate_aus = detector.detect_aus(gen)

            # gt_pose = detector.detect_facepose(gt, detected_faces=None, landmarks=None)
            # gt_pose = detector.detect_facepose(gt, detected_faces=None, landmarks=None)

            out_gt = detector.detect_image(gt)
            out_generat = detector.detect_image(gen)
            gt_pose = out_gt.facepose.values
            generate_pose = out_generat.facepose.values
            gt_aus = out_gt.aus.values
            generate_aus = out_generat.aus.values
            row,num = generate_aus.shape
            prmse=np.sqrt(np.power(gt_pose-generate_pose,2).sum()/3)
            if math.isnan(prmse):
                print(im)
                raise RuntimeError('NaN')
            PRMSE+=prmse
            generate_aus = generate_aus>0.5
            gt_aus = gt_aus>0.5
            rst = ~ (generate_aus^gt_aus)
            correct = rst.sum()
            AUCON+=(correct/num)
            counter+=1
        except Exception as e:
            print(e)
        try:
            source = Image.open(os.path.join(path,'source',im))
            generate = Image.open(os.path.join(path,'generate',im))

            # Get cropped and prewhitened image tensor
            img_cropped = mtcnn(source,save_path='src.jpg')
            # img_cropped = T.ToTensor()(source).cuda()
            # Calculate embedding (unsqueeze to add batch dimension)
            source_emb = resnet(img_cropped.unsqueeze(0))

            # Get cropped and prewhitened image tensor
            img_cropped = mtcnn(generate,save_path='dst.jpg')
            # img_cropped = T.ToTensor()(generate).cuda()
            # Calculate embedding (unsqueeze to add batch dimension)
            generate_emb = resnet(img_cropped.unsqueeze(0))
            CSIM+=torch.cosine_similarity(source_emb,generate_emb).item()
            csim_counter+=1
        except Exception as e:
            print(e)
    print(' PRMSE: {}, AUCON : {}, CSIM: {}'.format(PRMSE/counter, AUCON/counter,CSIM/csim_counter))
def mergeimgs(paths):
    pth = paths[0]
    imgps = os.listdir(pth)
    for i in tqdm(range(len(imgps))):
        imgp = imgps[i]
        cats = []
        for pth in paths:
            img = os.path.join(pth,imgp)
            img = cv2.imread(img)
            cats.append(img)
        img = np.vstack(cats)
        cv2.imwrite('FID/mergeimgs/{}.jpg'.format(i),img)

def create_animate_pair():
    f = open('./data/vox_cross_id_animate.csv','w',encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["source_frame","driving_video"])
    pairs = pd.read_csv('data/vox_cross_id_evaluation.csv')
    source = pairs['source'].tolist()
    driving = pairs['driving'].tolist()
    source_frames = []
    driving_videos = []

    for src, dst in zip(source,driving):
        video = os.path.dirname(dst).replace('vox1_frames','vox1')
        source_frames.append(src)
        driving_videos.append(video)
    source_frames = np.array(source_frames).reshape(-1,1)
    driving_videos = np.array(driving_videos).reshape(-1,1)
    content = np.concatenate((source_frames,driving_videos),1)
    csv_writer.writerows(content)
    f.close()

def merge_abla_imgs(paths):
    pth = paths[0]
    imgps = os.listdir(pth)
    for i in tqdm(range(len(imgps))):
        imgp = imgps[i]
        cats = []
        for pth in paths:
            img = os.path.join(pth,imgp)
            img = cv2.imread(img)
            cats.append(img)
        img = np.vstack(cats)
        cv2.imwrite('FID/abla/{}.jpg'.format(i),img)


def mergevideos():
    videos_path1 = 'animation'
    videos_path2 = '/data/fhongac/workspace/gitrepo/first-order-model/animation'
    videos = os.listdir(videos_path1)
    save_path = 'merge_animation'
    for vi in tqdm(videos):
        fomm = np.array(mimread('{}/{}'.format(videos_path2,vi),memtest=False))
        ours = np.array(mimread('{}/{}'.format(videos_path1,vi),memtest=False))
        reader = imageio.get_reader('{}/{}'.format(videos_path2,vi))
        fps = reader.get_meta_data()['fps']
        if len(fomm.shape) == 3:
            fomm = np.array([gray2rgb(frame) for frame in fomm])
        if fomm.shape[-1] == 4:
            fomm = fomm[..., :3]
        if len(ours.shape) == 3:
            ours = np.array([gray2rgb(frame) for frame in ours])
        if ours.shape[-1] == 4:
            ours = ours[..., :3]
        fomm = fomm[:,:,-256:,:]
        src_dst = ours[:,:,:512,:]
        ours = ours[:,:,-256:,:]
        merge = np.concatenate((src_dst,fomm,ours),2)
        imageio.mimsave('{}/{}'.format(save_path,vi), merge, fps=fps)

def extractFrames():
    videos_pairs = pd.read_csv('data/vox_cross_id_animate.csv')
    source = videos_pairs['source_frame'].tolist()
    driving = videos_pairs['driving_video'].tolist()
    frame_pairs = pd.read_csv('data/vox_cross_id_evaluation.csv')
    # source = videos_pairs['source_frame'].tolist()
    driving_frame = frame_pairs['driving'].tolist()
    concate = 'FID/video_cross_id'
    generate = 'FID/video_generate'
    videos = 'animation'
    for i, (src, dst,number) in tqdm(enumerate(zip(source,driving,driving_frame))):
        video = np.array(mimread('{}/{}.mp4'.format(videos,i),memtest=False))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        num = int(os.path.basename(number)[:7])
        video_array = img_as_float32(video)
        frame = (video_array[num]*255).astype(np.uint8)
        imsave('{}/{}.jpg'.format(concate,i),frame)
        imsave('{}/{}.jpg'.format(generate,i),frame[:,-256:,:])

class depth_network(nn.Module):
    def __init__(self):
        super(depth_network, self).__init__()
        self.depth_encoder = depth.ResnetEncoder(18, False).cuda()
        self.depth_decoder = depth.DepthDecoder(num_ch_enc=self.depth_encoder.num_ch_enc, scales=range(4)).cuda()
    def forward(self,x):
        outputs = self.depth_decoder(self.depth_encoder(x))
        return outputs
def viewNetworkStructure():
    network = depth_network().cuda()
    print(network)
    import hiddenlayer as h
    vis_graph = h.build_graph(network, torch.zeros([1,3,256,256]).cuda())   # 获取绘制图像的对象
    vis_graph.theme = h.graph.THEMES["blue"].copy()     # 指定主题颜色
    vis_graph.save("network_graph/depth_network.png")   # 保存图像的路径
   
def drawKPline():
    kp10 = [2.292730636,0.870793269,0.719648837]
    kp15 = [2.335680558,0.872849592,0.7229482939818654]
    kp20 = [2.268743373,0.882716346, 0.67557838]
    kp25 = [3.395401378,0.827983638,0.662669217]
    data = np.array([kp10,kp15,kp20,kp25])
    x=[0,1,2,3]
    
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    l1=plt.plot(x,data[:,0],'r--',label='PRMSE')
    l2=plt.plot(x,data[:,1],'g--',label='AUCON')
    l3=plt.plot(x,data[:,2],'b--',label='CSIM')

    plt.plot(x,data[:,0],'ro-',x,data[:,1],'g+-',x,data[:,2],'b^-')
    plt.grid(linestyle=':')
    # ax.tick_params(bottom=False)
    plt.xticks(x,["kp=10","kp=15","kp=20","kp=25"])  #去掉横坐标值
    # plt.yticks([])  #去掉纵坐标值
    # plt.setp(ax.get_xticklabels(), visible=False)
    # plt.setp(ax.get_yticklabels(), visible=False)
    plt.legend()
    plt.savefig('network_graph/kp.pdf')

def all_depth(path):
    imgs = os.listdir(path+'/gt')

    depth_encoder = depth.ResnetEncoder(18, False).cuda()
    depth_decoder = depth.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4)).cuda()
    loaded_dict_enc = torch.load('depth/models/weights_19/encoder.pth')
    loaded_dict_dec = torch.load('depth/models/weights_19/depth.pth')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()}
    depth_encoder.load_state_dict(filtered_dict_enc)
    depth_decoder.load_state_dict(loaded_dict_dec)
    depth_encoder.eval()
    depth_decoder.eval()
    for im in tqdm(imgs):
        driving = os.path.join(path,'gt',im)
        source = os.path.join(path,'generate',im)
    img = Image.open(path).convert('RGB').resize((256,256))
    tensor_img = T.ToTensor()(img).unsqueeze(0).cuda()
    outputs = depth_decoder(depth_encoder(tensor_img))
    disp = outputs[("disp", 0)]
    # Saving colormapped depth image
    disp_resized = torch.nn.functional.interpolate(disp, (256, 256), mode="bilinear", align_corners=False)
    disp_resized_np = disp_resized.squeeze().detach().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='rainbow')

    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    plt.axis('off')
    plt.imshow(colormapped_im)

    # plt.colorbar(mapper)
    plt.savefig(path+'.pdf')
    # plt.savefig(path+'.png')

    plt.clf()


def changevideos():
    # videos_path1 = 'animation'
    # videos_path2 = '/data/fhongac/workspace/gitrepo/first-order-model/animation'
    # videos = os.listdir(videos_path1)
    # save_path = 'merge_animation'
    # for vi in tqdm(videos):
    #     fomm = np.array(mimread('{}/{}'.format(videos_path2,vi),memtest=False))
    #     ours = np.array(mimread('{}/{}'.format(videos_path1,vi),memtest=False))
    #     reader = imageio.get_reader('{}/{}'.format(videos_path2,vi))
    #     fps = reader.get_meta_data()['fps']
    #     if len(fomm.shape) == 3:
    #         fomm = np.array([gray2rgb(frame) for frame in fomm])
    #     if fomm.shape[-1] == 4:
    #         fomm = fomm[..., :3]
    #     if len(ours.shape) == 3:
    #         ours = np.array([gray2rgb(frame) for frame in ours])
    #     if ours.shape[-1] == 4:
    #         ours = ours[..., :3]
    #     fomm = fomm[:,:,-256:,:]
    #     src_dst = ours[:,:,:512,:]
    #     ours = ours[:,:,-256:,:]
    #     merge = np.concatenate((src_dst,fomm,ours),2)
    #     imageio.mimsave('{}/{}'.format(save_path,vi), merge, fps=fps)
    # 155
    # path = 'merge_animation/155.mp4'
    # disp = '/data/fhongac/workspace/src/depthEstimate/7PbDDjXgYzU/id10287#bP0bKbQQlzc#003638#003940_disp.mp4'
    # osfv = '/data/fhongac/workspace/gitrepo/One-Shot_Free-View_Neural_Talking_Head_Synthesis/155.mp4'
    # save = 'FID/animation/155.mp4'

    # path = 'merge_animation/523.mp4'
    # disp = '/data/fhongac/workspace/src/depthEstimate/7PbDDjXgYzU/id10287#4oOmqI1myzY#000381#000729_disp.mp4'
    # osfv = '/data/fhongac/workspace/gitrepo/One-Shot_Free-View_Neural_Talking_Head_Synthesis/523.mp4'
    # save = 'FID/animation/523.mp4'

    # path = 'merge_animation/705.mp4'
    # disp = '/data/fhongac/workspace/src/depthEstimate/705.mp4'
    # osfv = '/data/fhongac/workspace/gitrepo/One-Shot_Free-View_Neural_Talking_Head_Synthesis/705.mp4'
    # save = 'FID/animation/705.mp4'

    # path = 'merge_animation/2062.mp4'
    # disp = '/data/fhongac/workspace/src/depthEstimate/2062.mp4'
    # osfv = '/data/fhongac/workspace/gitrepo/One-Shot_Free-View_Neural_Talking_Head_Synthesis/2062.mp4'
    # save = 'FID/animation/2062.mp4'

    # path = 'merge_animation/1841.mp4'
    # disp = '/data/fhongac/workspace/src/depthEstimate/1841.mp4'
    # osfv = '/data/fhongac/workspace/gitrepo/One-Shot_Free-View_Neural_Talking_Head_Synthesis/1841.mp4'
    # save = 'FID/animation/1841.mp4'

    path = 'merge_animation/1758.mp4'
    disp = '/data/fhongac/workspace/src/depthEstimate/1758.mp4'
    osfv = '/data/fhongac/workspace/gitrepo/One-Shot_Free-View_Neural_Talking_Head_Synthesis/1758.mp4'
    save = 'FID/animation/1758.mp4'
    
    video = np.array(mimread('{}'.format(path),memtest=False))
    reader = imageio.get_reader('{}'.format(path))
    fps = reader.get_meta_data()['fps']
    video = np.array([gray2rgb(frame) for frame in video])

    disp = np.array(mimread('{}'.format(disp),memtest=False))
    disp = np.array([gray2rgb(frame) for frame in disp])
    
    osfv = np.array(mimread('{}'.format(osfv),memtest=False))
    osfv = np.array([gray2rgb(frame) for frame in osfv])

    bz,h,w,c = video.shape
    up_video = np.concatenate((video[:,:,:int(w/2),:],disp),2)
    down_video = np.concatenate((video[:,:,int(w/2):int(w/4)*3,:],osfv,video[:,:,int(w/4)*3:,:]),2)

    up_zeros = np.ones((bz,20,256*3,3))*255
    mid_zeros = np.ones((bz,40,256*3,3))*255
    down_zeros = np.ones((bz,40,256*3,3))*255
    video = np.concatenate((up_zeros,up_video, mid_zeros, down_video,down_zeros),1)
    imageio.mimsave('{}'.format(save), video, fps=fps)

    print('aa')

def find_best_frame_video():
    import pandas as pd
    pairs = pd.read_csv('./data/celeV_cross_id_evaluation.csv')
    sources = pairs['source']
    drivings = pairs['driving']
    best_frame = []
    for i,(src,dri) in tqdm(enumerate(zip(sources,drivings))):
        source_image = imageio.imread(src)
        source_image = resize(source_image, (256, 256))[..., :3]
        vpath = os.path.dirname(dri)
        imgs = os.listdir(vpath)
        driving_video = []
        for j, im in enumerate(imgs):
            image = imageio.imread(os.path.join(vpath,im))
            driving_video.append(resize(image, (256, 256))[..., :3])
        idx = find_best_frame(source_image,driving_video)
        bf = os.path.join(vpath,imgs[idx])
        best_frame.append(bf)
    f = open('./data/celeV_cross_id_evaluation_best_frame.csv','w',encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["source","driving","best_frame"])
    sources = np.array(sources).reshape(-1,1)
    drivings = np.array(drivings).reshape(-1,1)
    best_frame = np.array(best_frame).reshape(-1,1)
    content = np.concatenate((sources,drivings,best_frame),1)
    csv_writer.writerows(content)
    f.close()
def find_best_frame(source, driving, cpu=False):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num

if __name__ == '__main__':
    # create_test_set('/data/fhongac/origDataset/vox1_frames/test')
    # concate_compared_results('FID/generate','/data/fhongac/workspace/gitrepo/first-order-model/FID/generate')
    # concate_compared_results('FID/concate','/data/fhongac/workspace/gitrepo/first-order-model/FID/concate')
    # create_cross_id_test_set('/data/fhongac/origDataset/vox1_frames/test')
    # render('ppt_figure/0000021.jpg')
    # depth_rgb('ppt_figure/293.jpg')
    # process_celeV('/data/fhongac/origDataset/CelebV')
    # compare()
    # evaluate_PRMSE_AUCON()  # CUDA_VISIBLE_DEVICES=7 python utils.py 
    # viewNetworkStructure()
    # aus('11.png')
    # mergeimgs(['/data/fhongac/workspace/gitrepo/first-order-model/checkpoint/vox_cross_id/concate',
    #         '/data/fhongac/workspace/gitrepo/One-Shot_Free-View_Neural_Talking_Head_Synthesis/checkpoint/vox_cross_id/concate',
    #         '/data/fhongac/workspace/src/parallel-fom-rgbd/log/vox-adv-256rgbd_kp_num15_rgbd_attnv2/vox_cross_id/concate'])    
    # create_animate_pair()
    # merge_abla_imgs(['/data/fhongac/workspace/gitrepo/first-order-model/FID/cross_id','log/vox-adv-256baseline/vox_cross_id/concate', 'log/vox-adv-256rgbd_kp_num15/vox_cross_id/concate','log/vox-adv-256rgbd_kp_num15_rgbd_attnv2_wo_D/vox_cross_id/concate','log/vox-adv-256rgbd_kp_num15_rgbd_attnv2/vox_cross_id/concate'])  
    # extractFrames()
    # mergevideos()
    # drawKPline()
    # changevideos()
    find_best_frame_video()

