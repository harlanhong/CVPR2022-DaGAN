import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d,SPADEResnetBlock
from modules.dense_motion import *
import pdb
from modules.AdaIN import calc_mean_std,adaptive_instance_normalization
from modules.dynamic_conv import Dynamic_conv2d
class SPADEGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        ic = 256
        cc = 4
        oc = 64
        norm_G = 'spadespectralinstance'
        label_nc = 3 + cc
        
        self.compress = nn.Conv2d(ic, cc, 3, padding=1)
        self.fc = nn.Conv2d(ic, 2 * ic, 3, padding=1)

        self.G_middle_0 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_1 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_2 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        # self.G_middle_3 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        # self.G_middle_4 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        # self.G_middle_5 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)

        self.up_0 = SPADEResnetBlock(2 * ic, ic, norm_G, label_nc)
        self.up_1 = SPADEResnetBlock(ic, oc, norm_G, label_nc)
        self.conv_img = nn.Conv2d(oc, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        
    def forward(self, feature, image):
        cp = self.compress(feature)
        seg = torch.cat((F.interpolate(cp, size=(image.shape[2], image.shape[3])), image), dim=1)   # 7, 256, 256
    
        x = feature      # 256, 64, 64
        x = self.fc(x)                # 512, 64, 64
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x = self.G_middle_2(x, seg)
        # x = self.G_middle_3(x, seg)
        # x = self.G_middle_4(x, seg)
        # x = self.G_middle_5(x, seg)
        x = self.up(x)                # 256, 128, 128
        x = self.up_0(x, seg)
        x = self.up(x)                # 64, 256, 256
        x = self.up_1(x, seg)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        # x = torch.tanh(x)
        x = F.sigmoid(x)
        
        return x

class DepthAwareAttention(nn.Module):
    """ depth-aware attention Layer"""
    def __init__(self,in_dim,activation):
        super(DepthAwareAttention,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,source,feat):
        """
            inputs :
                source : input feature maps( B X C X W X H) 256,64,64
                driving : input feature maps( B X C X W X H) 256,64,64
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = source.size()
        proj_query  = self.activation(self.query_conv(source)).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N) [bz,32,64,64]
        proj_key =  self.activation(self.key_conv(feat)).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.activation(self.value_conv(feat)).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        out = self.gamma*out + feat

        return out,attention     

#### main ####
class DepthAwareGenerator(nn.Module):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        super(DepthAwareGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        #source depth
        self.src_first = SameBlock2d(1, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        src_down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            src_down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.src_down_blocks = nn.ModuleList(src_down_blocks)

        # #driving depth
        # self.dst_first = SameBlock2d(1, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        # dst_down_blocks = []
        # for i in range(num_down_blocks):
        #     in_features = min(max_features, block_expansion * (2 ** i))
        #     out_features = min(max_features, block_expansion * (2 ** (i + 1)))
        #     dst_down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        # self.dst_down_blocks = nn.ModuleList(dst_down_blocks)

        self.AttnModule = DepthAwareAttention(out_features,nn.ReLU())

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation)

    def forward(self, source_image, kp_driving, kp_source, source_depth, driving_depth):
        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)

        src_out = self.src_first(source_depth)
        for i in range(len(self.src_down_blocks)):
            src_out = self.src_down_blocks[i](src_out)
        
        # dst_out = self.dst_first(driving_depth)
        # for i in range(len(self.down_blocks)):
        #     dst_out = self.dst_down_blocks[i](dst_out)
        
        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']
            output_dict['sparse_deformed'] = dense_motion['sparse_deformed']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            out = self.deform_input(out, deformation)

            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
                out = out * occlusion_map
            out,attention  = self.AttnModule(src_out,out)

            output_dict["deformed"] = self.deform_input(source_image, deformation)
            output_dict["attention"] = attention

        # Decoding part
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = F.sigmoid(out)

        output_dict["prediction"] = out

        return output_dict

class SPADEDepthAwareGenerator(nn.Module):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        super(SPADEDepthAwareGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        #source depth
        self.src_first = SameBlock2d(1, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        src_down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            src_down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.src_down_blocks = nn.ModuleList(src_down_blocks)

        # #driving depth
        # self.dst_first = SameBlock2d(1, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        # dst_down_blocks = []
        # for i in range(num_down_blocks):
        #     in_features = min(max_features, block_expansion * (2 ** i))
        #     out_features = min(max_features, block_expansion * (2 ** (i + 1)))
        #     dst_down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        # self.dst_down_blocks = nn.ModuleList(dst_down_blocks)

        self.AttnModule = DepthAwareAttention(out_features,nn.ReLU())
        self.decoder = SPADEGenerator()
        
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation)

    def forward(self, source_image, kp_driving, kp_source, source_depth, driving_depth):
        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)

        src_out = self.src_first(source_depth)
        for i in range(len(self.src_down_blocks)):
            src_out = self.src_down_blocks[i](src_out)
        
        # dst_out = self.dst_first(driving_depth)
        # for i in range(len(self.down_blocks)):
        #     dst_out = self.dst_down_blocks[i](dst_out)
        
        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']
            output_dict['sparse_deformed'] = dense_motion['sparse_deformed']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            out = self.deform_input(out, deformation)

            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
                out = out * occlusion_map
            
            out,attention  = self.AttnModule(src_out,out)

            deformed_image = self.deform_input(source_image, deformation)
            output_dict["deformed"] = deformed_image
            output_dict["attention"] = attention

            if occlusion_map is not None:
                if deformed_image.shape[2] != occlusion_map.shape[2] or deformed_image.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=deformed_image.shape[2:], mode='bilinear')
                deformed_image = deformed_image * occlusion_map
            
        out = self.decoder(out, deformed_image)

        # # Decoding part
        # out = self.bottleneck(out)
        # for i in range(len(self.up_blocks)):
        #     out = self.up_blocks[i](out)
        # out = self.final(out)
        # out = F.sigmoid(out)
        output_dict["prediction"] = out
        return output_dict
