import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.layers = list(self.model.children())[:-2]

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs[-3], outputs[-2], outputs[-1]


def dot_product_distance(feat1, feat2):
    feat1 = F.normalize(feat1, p=2, dim=1)
    feat2 = F.normalize(feat2, p=2, dim=1)
    mat = torch.matmul(feat1, feat2.T) / 0.1
    mat = F.softmax(mat, dim=-1)
    similarity = torch.diag(mat)
    return similarity


def js_div(p_output, q_output, get_softmax=True):
    """
    计算JS散度（Jensen-Shannon Divergence）作为相似度度量。
    参数:
        p_output: 张量，样本P的输出
        q_output: 张量，样本Q的输出
        get_softmax: 布尔值，是否应用softmax
    返回:
        两个分布的平均KL散度
    """
    KLDivLoss = nn.KLDivLoss(reduction='none')
    if get_softmax:
        p_output = F.softmax(p_output, dim=1)
        q_output = F.softmax(q_output, dim=1)

    mean_output = (p_output + q_output) / 2
    mean_output = mean_output.clamp(1e-6, 1 - 1e-6)
    p_output = p_output.clamp(1e-6, 1 - 1e-6)
    q_output = q_output.clamp(1e-6, 1 - 1e-6)
    log_mean_output = mean_output.log()

    part1 = KLDivLoss(log_mean_output, p_output).sum(dim=1)
    part2 = KLDivLoss(log_mean_output, q_output).sum(dim=1)
    return (part1 + part2) / 2


def js_distance(X, Y, win=7):
    """
    计算JS散度和L2距离的加权和。
    参数:
        X, Y: 输入特征图张量
        win: 窗口大小
    返回:
        加权后相似度分数
    """
    batch_size, chn_num, _, _ = X.shape
    # 重新调整输入特征图形状
    patch_x = X.shape[2] // win
    patch_y = X.shape[3] // win
    X_patch = X.view([batch_size, chn_num, win, patch_x, win, patch_y])
    Y_patch = Y.view([batch_size, chn_num, win, patch_x, win, patch_y])
    patch_num = patch_x * patch_y
    X_1D = X_patch.permute((0, 1, 3, 5, 2, 4)).contiguous().view([batch_size, -1, chn_num * patch_num])
    Y_1D = Y_patch.permute((0, 1, 3, 5, 2, 4)).contiguous().view([batch_size, -1, chn_num * patch_num])
    X_pdf = X_1D
    Y_pdf = Y_1D
    jsd = js_div(X_pdf, Y_pdf)
    L2 = ((X_1D - Y_1D) ** 2).sum(dim=1)
    w = (1 / (torch.sqrt(torch.exp((- 1 / (jsd + 10)))) * (jsd + 10) ** 2))
    final = jsd + L2 * w
    return final.mean(dim=1)


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1, ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, 1, 1, ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 3, 1, 1, ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )
        self.conv_in1 = nn.Conv2d(512, 512, 1, 1, 0)
        self.conv_in2 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.conv_in3 = nn.Conv2d(2048, 512, 1, 1, 0)

    def forward(self, x1, x2, x3):
        x1 = self.conv_in1(x1)
        x2 = self.conv_in2(x2)
        x3 = self.conv_in3(x3)
        x1 = self.transform(x1)
        x2 = self.transform(x2)
        x3 = self.transform(x3)
        return [x1, x2, x3]


class Decoder(nn.Module):
    def __init__(self, out_channels=1):
        super(Decoder, self).__init__()
        self.conv1 = reflect_conv(in_channels=512, kernel_size=3, out_channels=512, stride=1, pad=1)
        self.conv2 = reflect_conv(in_channels=512, kernel_size=3, out_channels=512, stride=1, pad=1)
        self.conv3 = reflect_conv(in_channels=512, kernel_size=3, out_channels=256, stride=1, pad=1)
        self.conv4 = reflect_conv(in_channels=256, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.conv5 = reflect_conv(in_channels=64, kernel_size=3, out_channels=out_channels, stride=1, pad=1)
        self.activate = nn.LeakyReLU()

    def forward(self, x):
        x = self.activate(self.conv1(x))
        x = self.activate(self.conv2(x))
        x = self.activate(self.conv3(x))
        x = self.activate(self.conv4(x))
        x = self.conv5(x)
        return (torch.tanh(x) + 1) / 2


# class Decoder(nn.Module):
#     def __init__(self, out_channels=1):
#         super(Decoder, self).__init__()
#         self.conv1 = reflect_conv(in_channels=512, kernel_size=3, out_channels=512, stride=1, pad=1)
#         self.conv2 = reflect_conv(in_channels=512, kernel_size=3, out_channels=512, stride=1, pad=1)
#         self.conv3 = reflect_conv(in_channels=512, kernel_size=3, out_channels=256, stride=1, pad=1)
#         self.conv4 = reflect_conv(in_channels=256, kernel_size=3, out_channels=64, stride=1, pad=1)
#         self.conv5 = reflect_conv(in_channels=64, kernel_size=3, out_channels=out_channels, stride=1, pad=1)
#         self.activate = nn.LeakyReLU()
#
#     def forward(self, x1, x2, x3):
#         x = self.activate(self.conv1(x3)) + x2
#         x = self.activate(self.conv2(x)) + x1
#         x = self.activate(self.conv3(x))
#         x = self.activate(self.conv4(x))
#         x = self.conv5(x)
#         return (torch.tanh(x) + 1) / 2


# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.conv1 = nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=4, stride=2, padding=1)
#         self.conv2 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)
#         self.conv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
#         self.conv4 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=1)
#         self.conv5 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1)
#         self.activate = nn.LeakyReLU()
#
#     def forward(self, x1, x2, x3):
#         x = self.activate(self.conv1(x3)) + x2
#         x = self.activate(self.conv2(x)) + x1
#         x = self.activate(self.conv3(x))
#         x = self.activate(self.conv4(x))
#         x = self.conv5(x)  # 将范围从[-1,1]转换为[0,1]
#         return x


# Soft K-Means 实现
class SoftKMeans(nn.Module):
    def __init__(self, n_clusters, feature_dim, temperature=1.0):
        super(SoftKMeans, self).__init__()
        self.num_clusters = n_clusters
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.centroids = nn.Parameter(torch.randn(n_clusters, feature_dim))

    def forward(self, x):
        x = x.view(x.shape[0], -1, self.feature_dim)
        distance_matrix = torch.cdist(x, self.centroids)
        soft_assignments = F.softmax(-distance_matrix / self.temperature, dim=-1)
        if self.training:
            return soft_assignments, self.compute_loss(x, soft_assignments)
        else:
            return soft_assignments, 0

    def compute_loss(self, x, soft_assignments):
        expanded_x = x.unsqueeze(2)
        # 3x784x128
        # -1x1x100
        expanded_centroids = self.centroids.unsqueeze(0).unsqueeze(0)
        distances = torch.sum((expanded_x - expanded_centroids) ** 2, dim=3)
        loss = torch.sum(soft_assignments * distances)
        return loss / x.size(0)


class reflect_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, pad=1):
        super(reflect_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=pad)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


def extract_luminance_structure_contrast(img):
    # 假设输入为RGB图像，首先转换为灰度图像提取亮度信息
    luminance = 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]

    # 使用Sobel算子提取结构信息（水平和垂直梯度的平方和开方）
    sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).to(img.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).to(img.device).unsqueeze(0).unsqueeze(0)

    grad_x = F.conv2d(luminance.unsqueeze(1), sobel_x, padding=1)
    grad_y = F.conv2d(luminance.unsqueeze(1), sobel_y, padding=1)
    structure = torch.sqrt(grad_x ** 2 + grad_y ** 2).squeeze(1)

    # 计算局部对比度（通过局部区域的亮度标准差）
    mean_luminance = F.avg_pool2d(luminance.unsqueeze(1), 3, stride=1, padding=1)
    contrast = torch.sqrt(F.avg_pool2d((luminance.unsqueeze(1) - mean_luminance) ** 2, 3, stride=1, padding=1)).squeeze(
        1)

    return luminance, structure, contrast
