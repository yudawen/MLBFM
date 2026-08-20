import torchvision
import torch
import torch.nn as nn
from lib.models.networks.transformer_models import Transformer_encoder
from lib.models.dance_lib.networks.dance.evolve813 import Dance
from lib.models.dance_lib.networks.dance.wrappers import Conv2d
import fvcore.nn.weight_init as weight_init
from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
import torch
import clip
from PIL import Image
from transformers import BitsAndBytesConfig

nms = torchvision.ops.nms
resnet = torchvision.models.resnet.resnet50(pretrained=True)
import numpy as np
import math
class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x
import torch
import torch.nn.functional as F




class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)

class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        if down_x != None:
            x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
class ResdiualBlock(nn.Module):
    """
    实现子module：Residual Block
    """
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResdiualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)
def extract_resnet_multiscale(clip_model, x):

    # CLIP视觉主干
    visual = clip_model.visual
    x=x.half()
    x = visual.conv1(x)
    x = visual.bn1(x)
    x = visual.relu1(x)
    x = visual.conv2(x)
    x = visual.bn2(x)
    x = visual.relu2(x)
    x = visual.conv3(x)
    x = visual.bn3(x)
    x = visual.relu3(x)
    x0 = visual.avgpool(x)

    x1 = visual.layer1(x0)   # low-level
    x2 = visual.layer2(x1)  # mid-low
    x3 = visual.layer3(x2)  # mid-high
    x4 = visual.layer4(x3)  # high-level
    return [x0, x1, x2, x3, x4]


def positional_encoding_2d(d_model, height, width):
  """
  :param d_model: dimension of the model
  :param height: height of the positions
  :param width: width of the positions
  :return: d_model*height*width position matrix
  """
  if d_model % 4 != 0:
    raise ValueError("Cannot use sin/cos positional encoding with "
                     "odd dimension (got dim={:d})".format(d_model))
  pe = torch.zeros(d_model, height, width)
  # Each dimension use half of d_model
  d_model = int(d_model / 2)
  div_term = torch.exp(torch.arange(0., d_model, 2) *
                       -(math.log(10000.0) / d_model))
  pos_w = torch.arange(0., width).unsqueeze(1)
  pos_h = torch.arange(0., height).unsqueeze(1)
  pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
  pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
  pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
  pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
  return pe

class ResNet50(nn.Module):
    DEPTH = 6
    def __init__(self, class_num=10):
        super().__init__()
        self.class_num=class_num
        print('using the resnet 50 backbone!')
        # model_name=r"llava-hf/llava-1.5-7b-hf",
        device = 'cuda'

        self.heads = {'hm':1,'wh':2,'reg':2}
        self.device = 'cuda'

        # ===== 1. processor =====
        model_name = "llava-hf/llava-1.5-7b-hf"
        self.processor = AutoProcessor.from_pretrained(model_name)

        # ===== 2. 添加 SEG token =====
        self.processor.tokenizer.add_tokens(["[SEG]"], special_tokens=True)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        # ===== 3. model =====
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16

        ).to(device)
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        # resize token embedding
        self.model.resize_token_embeddings(len(self.processor.tokenizer))

        # ===== 4. 冻结参数 =====
        for p in self.model.parameters():
            p.requires_grad = False

        # ===== 5. LoRA =====
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, lora_config)

        # ===== 6. SEG token id =====

        seg_tokens = [f"[SEG{i}]" for i in range(class_num)]
        self.processor.tokenizer.add_tokens(seg_tokens)
        self.model.resize_token_embeddings(len(self.processor.tokenizer))

        # 记录 token id
        self.seg_token_ids = [
            self.processor.tokenizer.convert_tokens_to_ids(t)
            for t in seg_tokens
        ]





        # 加载预训练的 CLIP 模型，指定 "RN50" 表示使用 ResNet50 作为图像编码器
        self.clip_resnet50, self.clip_preprocess = clip.load("RN50", device="cuda" if torch.cuda.is_available() else "cpu")
        # print(self.clip_preprocess)
        for p in self.clip_resnet50.parameters():
            p.requires_grad = False




        self.adapter4 = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1, groups=1024),
            nn.Conv2d(1024, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.adapter5 = nn.Sequential(
            nn.Conv2d(2048, 2048, 3, padding=1, groups=2048),
            nn.Conv2d(2048, 2048, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
        seg_up_blocks = []

        self.seg_bridge = Bridge(2048, 1024)
        seg_up_blocks.append(UpBlockForUNetWithResNet50(in_channels=512+1024, out_channels=768, up_conv_in_channels=1024, up_conv_out_channels=512))
        seg_up_blocks.append(UpBlockForUNetWithResNet50(in_channels=512+512, out_channels=512, up_conv_in_channels=768, up_conv_out_channels=512))
        seg_up_blocks.append(UpBlockForUNetWithResNet50(in_channels=256+256, out_channels=256, up_conv_in_channels=512, up_conv_out_channels=256))

        self.gcn = Dance()
        head_conv=256


        self.seg_up_blocks=nn.ModuleList(seg_up_blocks)

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(head_conv, head_conv,kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=1, stride=1,
                              padding=1 // 2, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(head_conv, classes,
                               kernel_size=1, stride=1,
                               padding=1 // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)


        self.pos_trans_ori = nn.Sequential(nn.Linear(4096, 4096*2), nn.GELU())

        self.pos_trans = nn.Sequential(nn.Linear(4096, 1024), nn.GELU(), nn.Linear(1024, 256),nn.LayerNorm(256))
        self.pos_trans1 = nn.Sequential(nn.Linear(4096, 4096), nn.GELU())
        self.pos_trans2 =nn.Sequential( nn.Linear(4096, 4096), nn.LayerNorm(4096))

        # 动量系数
        self.m = 0.95  # 0.9~0.99 都可

    def remove_cls(self, feat):
        return feat[:, 1:, :]  # 去掉 CLS

    def tokens_to_map(self, feat):
        B, N, D = feat.shape
        H = W = int(N ** 0.5)
        return feat.transpose(1, 2).reshape(B, D, H, W)
    def positional_encoding_2d(self,d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe
    def get_pixel_features(self,image_size, d_pe=128):
        all_pe = self.positional_encoding_2d(d_pe, image_size, image_size)
        pixels_x = np.arange(0, image_size)
        pixels_y = np.arange(0, image_size)

        xv, yv = np.meshgrid(pixels_x, pixels_y)
        all_pixels = list()
        for i in range(xv.shape[0]):
            pixs = np.stack([xv[i], yv[i]], axis=-1)
            all_pixels.append(pixs)
        pixels = np.stack(all_pixels, axis=0)

        pixel_features = all_pe[:, pixels[:, :, 1], pixels[:, :, 0]]
        pixel_features = pixel_features.permute(1, 2, 0)
        return pixels, pixel_features
    def forward(self, x,prompt=None,category=None):

        # segmentation part
        seg_pre_pools = dict()
        seg_pre_pools[f"layer_0"] = x
        multi_scale = extract_resnet_multiscale(self.clip_resnet50, x)

        multi_scale[3] = multi_scale[3].float() + self.adapter4(multi_scale[3].float())
        multi_scale[4] = multi_scale[4].float() + self.adapter5(multi_scale[4].float())

        seg_x=multi_scale[4].float()
        seg_pre_pools[f"layer_1"] = multi_scale[0].float()
        seg_pre_pools[f"layer_2"] = multi_scale[1].float()
        seg_pre_pools[f"layer_3"] = multi_scale[2].float()
        seg_pre_pools[f"layer_4"] = multi_scale[3].float()
        seg_pre_pools[f"layer_5"]=seg_x


        if 1:#category is not None:
            category_num=self.class_num


            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)

            # 1. undo normalize
            x = x * std + mean

            # 2. undo /255
            x = x * 255.0

            # 3. clamp
            x = x.clamp(0, 255)

            xpli = [
                Image.fromarray(img.transpose(1,2,0).astype('uint8'))
                for img in x.detach().cpu().numpy()
            ]


            # 2️⃣ processor
            prompt_=[prompt[i]+category[i] for i in range(len(prompt))]

            inputs = self.processor(
                text=prompt_,
                images=xpli,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            input_ids = inputs["input_ids"]

            # 3️⃣ labels
            labels = input_ids.clone()

            for i in range(len(prompt)):
                prompt_ids = self.processor.tokenizer(prompt[i]).input_ids
                labels[i, :len(prompt_ids)] = -100

            # 4️⃣ forward
            outputs = self.model(**inputs, labels=labels, output_hidden_states=True)

            loss_llm = outputs.loss

            # 5️⃣ seg embeddings
            hidden_states = outputs.hidden_states[-1]
            B, L, C = hidden_states.shape

            # -------------------------
            # 4️⃣ 精确提取每个 seg_i
            # -------------------------
            seg_embeddings = []

            for i in range(category_num):
                token_id = self.seg_token_ids[i]
                mask = (input_ids == token_id)  # [B, L]


                emb_i = hidden_states[mask]  # [B, C]
                seg_embeddings.append(emb_i)

            seg_embeddings_ori = torch.stack(seg_embeddings, dim=1).float()  # [B, K, C]
            seg_embeddings_ori=self.pos_trans_ori(seg_embeddings_ori)

            seg_embeddings=self.pos_trans(seg_embeddings_ori[:,:,:4096])
            seg_embeddings_ori=self.pos_trans1(seg_embeddings_ori[:,:,4096:])
            pe = positional_encoding_2d(20, 64, 64)[:category_num,:,:]
            geo_feature = pe.unsqueeze(0).expand(seg_embeddings_ori.size(0), category_num, 64, 64).cuda()

            geo_feature = geo_feature.view(seg_embeddings_ori.size())

            seg_embeddings_ori = seg_embeddings_ori + geo_feature
            seg_embeddings_ori=self.pos_trans2(seg_embeddings_ori)


            seg_embeddings = seg_embeddings.view(B, category_num, -1).contiguous()
            seg_embeddings_ori= seg_embeddings_ori.view(B, category_num, -1).contiguous()
        else:
            seg_embeddings=None
            seg_embeddings_ori=None

            loss_llm=torch.tensor(0.0)


        seg_x = self.seg_bridge(seg_pre_pools[f"layer_5"])
        xs_={}

        for i, block in enumerate(self.seg_up_blocks[:3], 1):
            key = f"layer_{ResNet50.DEPTH - 1 - i}"
            seg_x = block(seg_x, seg_pre_pools[key])


            xs_.update({str(3-i): seg_x})




        xs = {"0": xs_["0"], "1": xs_["1"], "2": xs_["2"]}  #

        all_feats = {'layer0': seg_pre_pools[f"layer_1"], 'layer1': seg_pre_pools[f"layer_2"],
                     'layer2': seg_pre_pools[f"layer_3"], 'layer3': seg_pre_pools[f"layer_4"],
                     'x_original': seg_pre_pools[f"layer_0"]}
        mask = torch.zeros(seg_pre_pools[f"layer_2"].shape)[:, 0, :, :].to(seg_pre_pools[f"layer_0"].device)

        z={}
        for head in self.heads:
            z[head] = self.__getattr__(head)(xs_["0"])



        return [z],seg_x,xs, mask, all_feats,[seg_embeddings,seg_embeddings_ori],loss_llm*0.1

def get_pose_net(num_layers, heads, head_conv):

  model = ResNet50()

  return model

