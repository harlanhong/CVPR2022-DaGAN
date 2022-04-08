import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from frames_dataset import PairedDataset
from logger import Logger, Visualizer
import imageio
from scipy.spatial import ConvexHull
import numpy as np
import depth
from sync_batchnorm import DataParallelWithCallback


def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])
    return kp_new


def animate(config, generator, kp_detector, checkpoint, log_dir, dataset,opt):
    log_dir = os.path.join(log_dir, 'animation')
    png_dir = os.path.join(log_dir, 'png')
    animate_params = config['animate_params']

    dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=animate_params['num_pairs'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError("Checkpoint should be specified for mode='animate'.")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    
    depth_encoder = depth.ResnetEncoder(18, False).cuda()
    depth_decoder = depth.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4)).cuda()
    loaded_dict_enc = torch.load('depth/models/weights_19/encoder.pth')
    loaded_dict_dec = torch.load('depth/models/weights_19/depth.pth')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()}
    depth_encoder.load_state_dict(filtered_dict_enc)
    depth_decoder.load_state_dict(loaded_dict_dec)
    depth_decoder.eval()
    depth_encoder.eval()
    generator.eval()
    kp_detector.eval()

    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            predictions = []
            visualizations = []

            driving_video = x['driving_video'].cuda()
            source_frame = x['source_video'][:, :, 0, :, :].cuda()
            
            outputs = depth_decoder(depth_encoder(source_frame))
            depth_source = outputs[("disp", 0)]
            outputs = depth_decoder(depth_encoder(driving_video[:, :, 0]))
            depth_driving = outputs[("disp", 0)]

            source = torch.cat((source_frame,depth_source),1)
            driving = torch.cat((driving_video[:, :, 0],depth_driving),1)

            kp_source = kp_detector(source)
            kp_driving_initial = kp_detector(driving)

            for frame_idx in range(driving_video.shape[2]):
                driving_frame = driving_video[:, :, frame_idx].cuda()
                outputs = depth_decoder(depth_encoder(driving_frame))
                depth_map = outputs[("disp", 0)]
                driving = torch.cat((driving_frame,depth_map),1)
                kp_driving = kp_detector(driving)
                
                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                       kp_driving_initial=kp_driving_initial, **animate_params['normalization_params'])
                out = generator(source_frame, kp_source=kp_source, kp_driving=kp_norm)

                out['kp_driving'] = kp_driving
                out['kp_source'] = kp_source
                out['kp_norm'] = kp_norm

                del out['sparse_deformed']

                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                visualization = Visualizer(**config['visualizer_params']).visualize(source=source_frame,
                                                                                    driving=driving_frame, out=out)
                visualization = visualization
                visualizations.append(visualization)

            predictions = np.concatenate(predictions, axis=1)
            result_name = "-".join([x['driving_name'][0], x['source_name'][0]])
            imageio.imsave(os.path.join(png_dir, result_name + '.png'), (255 * predictions).astype(np.uint8))

            image_name = result_name + animate_params['format']
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations)

