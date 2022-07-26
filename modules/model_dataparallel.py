from torch import nn
import torch
import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d, make_coordinate_grid
from torchvision import models
import numpy as np
from torch.autograd import grad
import pdb
import depth

class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class Transform:
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params,opt):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()
        self.opt = opt
        self.loss_weights = train_params['loss_weights']

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()
        self.depth_encoder = depth.ResnetEncoder(50, False).cuda()
        self.depth_decoder = depth.DepthDecoder(num_ch_enc=self.depth_encoder.num_ch_enc, scales=range(4)).cuda()
        loaded_dict_enc = torch.load('depth/models/depth_face_model_Voxceleb2_10w/encoder.pth',map_location='cpu')
        loaded_dict_dec = torch.load('depth/models/depth_face_model_Voxceleb2_10w/depth.pth',map_location='cpu')
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.depth_encoder.state_dict()}
        self.depth_encoder.load_state_dict(filtered_dict_enc)
        self.depth_decoder.load_state_dict(loaded_dict_dec)
        self.set_requires_grad(self.depth_encoder, False) 
        self.set_requires_grad(self.depth_decoder, False) 
        self.depth_decoder.eval()
        self.depth_encoder.eval()
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    def forward(self, x):
        depth_source = None
        depth_driving = None
        outputs = self.depth_decoder(self.depth_encoder(x['source']))
        depth_source = outputs[("disp", 0)]
        outputs = self.depth_decoder(self.depth_encoder(x['driving']))
        depth_driving = outputs[("disp", 0)]
        
        if self.opt.use_depth:
            kp_source = self.kp_extractor(depth_source)
            kp_driving = self.kp_extractor(depth_driving)
        elif self.opt.rgbd:
            source = torch.cat((x['source'],depth_source),1)
            driving = torch.cat((x['driving'],depth_driving),1)
            kp_source = self.kp_extractor(source)
            kp_driving = self.kp_extractor(driving)
        else:
            kp_source = self.kp_extractor(x['source'])
            kp_driving = self.kp_extractor(x['driving'])
        generated = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving, source_depth = depth_source, driving_depth = depth_driving)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})
        loss_values = {}
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])
        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
                loss_values['perceptual'] = value_total

        if self.loss_weights['generator_gan'] != 0:

            discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))

            discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total

        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])
            if self.opt.use_depth:
                outputs = self.depth_decoder(self.depth_encoder(transformed_frame))
                depth_transform = outputs[("disp", 0)]
                transformed_kp = self.kp_extractor(depth_transform)
            elif self.opt.rgbd:
                outputs = self.depth_decoder(self.depth_encoder(transformed_frame))
                depth_transform = outputs[("disp", 0)]
                transform_img = torch.cat((transformed_frame,depth_transform),1)
                transformed_kp = self.kp_extractor(transform_img)
            else:
                transformed_kp = self.kp_extractor(transformed_frame)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp

            ## Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                value = torch.abs(kp_driving['value'] - transform.warp_coordinates(transformed_kp['value'])).mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

            ## jacobian loss part
            if self.loss_weights['equivariance_jacobian'] != 0:
                jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp['value']),
                                                    transformed_kp['jacobian'])

                normed_driving = torch.inverse(kp_driving['jacobian'])
                normed_transformed = jacobian_transformed
                value = torch.matmul(normed_driving, normed_transformed)

                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                value = torch.abs(eye - value).mean()
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value


        if self.loss_weights['kp_distance']:
            bz,num_kp,kp_dim = kp_source['value'].shape
            sk = kp_source['value'].unsqueeze(2)-kp_source['value'].unsqueeze(1)
            dk = kp_driving['value'].unsqueeze(2)-kp_driving['value'].unsqueeze(1)
            source_dist_loss = (-torch.sign((torch.sqrt((sk*sk).sum(-1)+1e-8)+torch.eye(num_kp).cuda()*0.2)-0.2)+1).mean()
            driving_dist_loss = (-torch.sign((torch.sqrt((dk*dk).sum(-1)+1e-8)+torch.eye(num_kp).cuda()*0.2)-0.2)+1).mean()
            # driving_dist_loss = (torch.sign(1-(torch.sqrt((dk*dk).sum(-1)+1e-8)+torch.eye(num_kp).cuda()))+1).mean()
            value_total = self.loss_weights['kp_distance']*(source_dist_loss+driving_dist_loss)
            loss_values['kp_distance'] = value_total
        if self.loss_weights['kp_prior']:
            bz,num_kp,kp_dim = kp_source['value'].shape
            sk = kp_source['value'].unsqueeze(2)-kp_source['value'].unsqueeze(1)
            dk = kp_driving['value'].unsqueeze(2)-kp_driving['value'].unsqueeze(1)
            dis_loss = torch.relu(0.1-torch.sqrt((sk*sk).sum(-1)+1e-8))+torch.relu(0.1-torch.sqrt((dk*dk).sum(-1)+1e-8))
            bs,nk,_=kp_source['value'].shape
            scoor_depth = F.grid_sample(depth_source,kp_source['value'].view(bs,1,nk,-1))
            dcoor_depth = F.grid_sample(depth_driving,kp_driving['value'].view(bs,1,nk,-1))
            sd_loss = torch.abs(scoor_depth.mean(-1,keepdim=True) - kp_source['value'].view(bs,1,nk,-1)).mean()
            dd_loss = torch.abs(dcoor_depth.mean(-1,keepdim=True) - kp_driving['value'].view(bs,1,nk,-1)).mean()
            value_total = self.loss_weights['kp_distance']*(dis_loss+sd_loss+dd_loss)
            loss_values['kp_distance'] = value_total


        if self.loss_weights['kp_scale']:
            bz,num_kp,kp_dim = kp_source['value'].shape
            if self.opt.rgbd:
                outputs = self.depth_decoder(self.depth_encoder(generated['prediction']))
                depth_pred = outputs[("disp", 0)]
                pred = torch.cat((generated['prediction'],depth_pred),1)
                kp_pred = self.kp_extractor(pred)
            elif self.opt.use_depth:
                outputs = self.depth_decoder(self.depth_encoder(generated['prediction']))
                depth_pred = outputs[("disp", 0)]
                kp_pred = self.kp_extractor(depth_pred)
            else:
                kp_pred = self.kp_extractor(generated['prediction'])

            pred_mean = kp_pred['value'].mean(1,keepdim=True)
            driving_mean = kp_driving['value'].mean(1,keepdim=True)
            pk = kp_source['value']-pred_mean
            dk = kp_driving['value']- driving_mean
            pred_dist_loss = torch.sqrt((pk*pk).sum(-1)+1e-8)
            driving_dist_loss = torch.sqrt((dk*dk).sum(-1)+1e-8)
            scale_vec = driving_dist_loss/pred_dist_loss
            bz,n = scale_vec.shape
            value = torch.abs(scale_vec[:,:n-1]-scale_vec[:,1:]).mean()
            value_total = self.loss_weights['kp_scale']*value
            loss_values['kp_scale'] = value_total
        if self.loss_weights['depth_constraint']:
            bz,num_kp,kp_dim = kp_source['value'].shape
            outputs = self.depth_decoder(self.depth_encoder(generated['prediction']))
            depth_pred = outputs[("disp", 0)]
            value_total = self.loss_weights['depth_constraint']*torch.abs(depth_driving-depth_pred).mean()
            loss_values['depth_constraint'] = value_total
        return loss_values, generated



class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'].detach())

        kp_driving = generated['kp_driving']
        discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
        discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))

        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            value_total += self.loss_weights['discriminator_gan'] * value.mean()
        loss_values['disc_gan'] = value_total

        return loss_values
