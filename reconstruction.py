import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from logger import Logger, Visualizer
import numpy as np
import imageio
from sync_batchnorm import DataParallelWithCallback
import depth


def reconstruction(config, generator, kp_detector, checkpoint, log_dir, dataset):
    png_dir = os.path.join(log_dir, 'reconstruction/png')
    log_dir = os.path.join(log_dir, 'reconstruction')

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    loss_list = []
    if torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

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
        if config['reconstruction_params']['num_videos'] is not None:
            if it > config['reconstruction_params']['num_videos']:
                break
        with torch.no_grad():
            predictions = []
            visualizations = []
            if torch.cuda.is_available():
                x['video'] = x['video'].cuda()

            outputs = depth_decoder(depth_encoder(x['video'][:, :, 0]))
            depth_source = outputs[("disp", 0)]
            source_rgbd = torch.cat((x['video'][:, :, 0],depth_source),1)

            kp_source = kp_detector(source_rgbd)
            
            for frame_idx in range(x['video'].shape[2]):
                source = x['video'][:, :, 0]

                driving = x['video'][:, :, frame_idx]
                outputs = depth_decoder(depth_encoder(driving))
                depth_driving = outputs[("disp", 0)]
                driving_rgbd = torch.cat((driving,depth_driving),1)

                kp_driving = kp_detector(driving_rgbd)
                out = generator(source, kp_source=kp_source, kp_driving=kp_driving)
                out['kp_source'] = kp_source
                out['kp_driving'] = kp_driving
                del out['sparse_deformed']
                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                visualization = Visualizer(**config['visualizer_params']).visualize(source=source,
                                                                                    driving=driving, out=out)
                visualizations.append(visualization)

                loss_list.append(torch.abs(out['prediction'] - driving).mean().cpu().numpy())

            predictions = np.concatenate(predictions, axis=1)
            imageio.imsave(os.path.join(png_dir, x['name'][0] + '.png'), (255 * predictions).astype(np.uint8))

            image_name = x['name'][0] + config['reconstruction_params']['format']
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations)

    print("Reconstruction loss: %s" % np.mean(loss_list))