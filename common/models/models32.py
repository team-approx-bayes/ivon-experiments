# models for 3x32x32 image inputs, e.g., CIFAR-10 / CIFAR-100
from typing import Tuple
import torch
import torch.nn as nn
from . import vgg_frn
from . import resnet_frn
from . import densenet_frn
from .frn import FilterResponseNorm


# cf. https://github.com/pytorch/pytorch/issues/3867#issuecomment-349279036
# should replicate tensorflow "SAME" padding behavior
def conv_same_padding(
    in_size: int, kernel: int, stride: int = 1, dilation: int = 1
) -> Tuple[int, int]:
    effective_filter_size = (kernel - 1) * dilation + 1
    out_size = (in_size + stride - 1) // stride
    padding_needed = max(
        0, (out_size - 1) * stride + effective_filter_size - in_size
    )
    if padding_needed % 2 == 0:
        padding_left = padding_needed // 2
        padding_right = padding_needed // 2
    else:
        padding_left = (padding_needed - 1) // 2
        padding_right = (padding_needed + 1) // 2
    return padding_left, padding_right


class ResnetBlock(nn.Module):
    def __init__(
        self,
        normalization_layer,
        input_size,
        num_filters,
        kernel_size=3,
        strides=1,
        activation=torch.nn.Identity,
        use_bias=True,
    ):
        super(ResnetBlock, self).__init__()
        # input size = C, H, W
        p0, p1 = conv_same_padding(input_size[2], kernel_size, strides)
        # height padding
        p2, p3 = conv_same_padding(input_size[1], kernel_size, strides)
        self.pad1 = torch.nn.ZeroPad2d((p0, p1, p2, p3))
        self.conv1 = torch.nn.Conv2d(
            input_size[0],
            num_filters,
            kernel_size=kernel_size,
            stride=strides,
            padding=0,
            bias=use_bias,
        )
        self.norm1 = normalization_layer(num_filters)
        self.activation1 = activation()

    def forward(self, x):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.activation1(out)

        return out


class StackedResnetBlock(nn.Module):
    def __init__(
        self,
        normalization_layer,
        num_filters,
        input_num_filters,
        stack,
        res_block,
        activation,
        use_bias,
        input_size,
    ):
        super(StackedResnetBlock, self).__init__()
        self.stack = stack
        self.res_block = res_block
        spatial_out = input_size // (2**stack)
        if stack > 0 and res_block == 0:  # first layer but not first stack
            strides = 2  # downsample
        else:
            strides = 1
        spatial_in = spatial_out * strides

        self.res1 = ResnetBlock(
            normalization_layer=normalization_layer,
            num_filters=num_filters,
            input_size=(input_num_filters, spatial_in, spatial_in),
            strides=strides,
            activation=activation,
            use_bias=use_bias,
        )
        self.res2 = ResnetBlock(
            normalization_layer=normalization_layer,
            num_filters=num_filters,
            input_size=(num_filters, spatial_out, spatial_out),
            use_bias=use_bias,
        )
        if stack > 0 and res_block == 0:  # first layer but not first stack
            # linear projection residual shortcut to match changed dims
            self.res3 = ResnetBlock(
                normalization_layer=normalization_layer,
                num_filters=num_filters,
                input_size=(input_num_filters, spatial_in, spatial_in),
                strides=strides,
                kernel_size=1,
                use_bias=use_bias,
            )

        self.activation1 = activation()

    def forward(self, x):
        y = self.res1(x)
        y = self.res2(y)
        if self.stack > 0 and self.res_block == 0:
            x = self.res3(x)
        out = self.activation1(x + y)
        return out


class MakeResnetFn(nn.Module):
    def __init__(
        self,
        num_classes,
        depth,
        normalization_layer,
        width=16,
        use_bias=True,
        activation=torch.nn.Identity,
        input_size=32,
    ):
        super(MakeResnetFn, self).__init__()
        self.num_res_blocks = (depth - 2) // 6
        self.normalization_layer = normalization_layer
        self.activation = activation
        self.use_bias = use_bias
        self.width = width
        self.input_size = input_size
        if (depth - 2) % 6 != 0:
            raise ValueError("depth must be 6n+2 (e.g. 20, 32, 44).")

        # first res_layer
        self.layer1 = ResnetBlock(
            normalization_layer=normalization_layer,
            num_filters=width,
            input_size=(3, input_size, input_size),
            kernel_size=3,
            strides=1,
            activation=torch.nn.Identity,
            use_bias=True,
        )
        # stacks
        self.stacks = self._make_res_block()
        # avg pooling
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        # torch.nn.AvgPool2d(kernel_size=(8, 8), stride=8, padding=0)
        # linear layer
        self.linear1 = nn.Linear(4 * width, num_classes)

    def forward(self, x):
        # first res_layer
        out = self.layer1(x)  # shape out torch.Size([5, 16, 32, 32])
        out = self.stacks(out)
        out = self.avgpool1(out)
        out = torch.flatten(out, start_dim=1)
        logits = self.linear1(out)
        return logits

    def _make_res_block(self):
        layers = list()
        num_filters = self.width
        input_num_filters = num_filters
        for stack in range(3):
            for res_block in range(self.num_res_blocks):
                layers.append(
                    StackedResnetBlock(
                        self.normalization_layer,
                        num_filters,
                        input_num_filters,
                        stack,
                        res_block,
                        self.activation,
                        self.use_bias,
                        self.input_size,
                    )
                )
                input_num_filters = num_filters
            num_filters *= 2
        return nn.Sequential(*layers)


def make_resnet20_frn_fn(data_info, activation=torch.nn.Identity):
    num_classes = data_info["num_classes"]
    input_size = data_info.get("input_size", 32)
    print(activation)
    return MakeResnetFn(
        num_classes,
        depth=20,
        normalization_layer=FilterResponseNorm,
        activation=activation,
        input_size=input_size,
    )


class PreResnetBlock(nn.Module):
    def __init__(
        self,
        normalization_layer,
        input_size,
        num_filters,
        kernel_size=3,
        strides=1,
        activation=torch.nn.Identity,
        use_bias=True,
    ):
        super().__init__()
        # input size = C, H, W
        p0, p1 = conv_same_padding(input_size[2], kernel_size, strides)
        # height padding
        p2, p3 = conv_same_padding(input_size[1], kernel_size, strides)
        self.pad1 = torch.nn.ZeroPad2d((p0, p1, p2, p3))
        self.conv1 = torch.nn.Conv2d(
            input_size[0],
            num_filters,
            kernel_size=kernel_size,
            stride=strides,
            padding=0,
            bias=use_bias,
        )
        self.norm1 = normalization_layer(num_filters)
        self.activation1 = activation()

    def forward(self, x):
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.pad1(x)
        x = self.conv1(x)

        return x


class StackedPreResnetBlock(nn.Module):
    def __init__(
        self,
        normalization_layer,
        num_filters,
        input_num_filters,
        stack,
        res_block,
        activation,
        use_bias,
        input_size,
    ):
        super().__init__()
        self.stack = stack
        self.res_block = res_block
        spatial_out = input_size // (2**stack)
        if stack > 0 and res_block == 0:  # first layer but not first stack
            strides = 2  # downsample
        else:
            strides = 1
        spatial_in = spatial_out * strides
        if stack > 0 and res_block == 0:  # first layer but not first stack
            # linear projection residual shortcut to match changed dims
            self.norm = normalization_layer(input_num_filters)
            self.activation = activation()
            self.res1 = PreResnetBlock(
                normalization_layer=torch.nn.Identity,
                num_filters=num_filters,
                input_size=(input_num_filters, spatial_in, spatial_in),
                strides=strides,
                use_bias=use_bias,
            )
            self.res3 = PreResnetBlock(
                normalization_layer=torch.nn.Identity,
                num_filters=num_filters,
                input_size=(input_num_filters, spatial_in, spatial_in),
                strides=strides,
                kernel_size=1,
                use_bias=use_bias,
            )
        else:
            self.norm = torch.nn.Identity()
            self.activation = torch.nn.Identity()
            self.res1 = PreResnetBlock(
                normalization_layer=normalization_layer,
                num_filters=num_filters,
                input_size=(input_num_filters, spatial_in, spatial_in),
                strides=strides,
                activation=activation,
                use_bias=use_bias,
            )
            self.res3 = torch.nn.Identity()

        self.res2 = PreResnetBlock(
            normalization_layer=normalization_layer,
            num_filters=num_filters,
            input_size=(num_filters, -1, -1),
            use_bias=use_bias,
            activation=activation,
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        r = self.res1(x)
        r = self.res2(r)
        s = self.res3(x)
        out = s + r
        return out


class MakePreResnetFn(nn.Module):
    def __init__(
        self,
        num_classes,
        depth,
        normalization_layer,
        width=24,
        use_bias=True,
        activation=torch.nn.Identity,
        input_size=32,
    ):
        super().__init__()
        self.num_res_blocks = (depth - 2) // 6
        self.normalization_layer = normalization_layer
        self.activation = activation
        self.use_bias = use_bias
        self.width = width
        self.input_size = input_size
        if (depth - 2) % 6 != 0:
            raise ValueError("depth must be 6n+2 (e.g. 20, 32, 44).")

        # first res_layer
        self.layer1 = PreResnetBlock(
            normalization_layer=torch.nn.Identity,
            num_filters=width,
            input_size=(3, -1, -1),
            kernel_size=3,
            strides=1,
            activation=torch.nn.Identity,
            use_bias=True,
        )
        self.norm1 = normalization_layer(width)
        self.activation1 = activation()
        # stacks
        self.stacks = self._make_res_block()
        # last norm + activaiton
        self.norm2 = normalization_layer(width * 4)
        self.activation2 = activation()
        # avg pooling
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        # self.avgpool1 = torch.nn.AvgPool2d(
        #    kernel_size=(8, 8), stride=8, padding=0)
        # linear layer
        self.linear1 = nn.Linear(width * 4, num_classes)

    def forward(self, x):
        # first res_layer
        out = self.layer1(x)  # shape out torch.Size([5, 16, 32, 32])
        out = self.norm1(out)
        out = self.activation1(out)
        out = self.stacks(out)
        out = self.norm2(out)
        out = self.activation2(out)
        out = self.avgpool1(out)
        out = torch.flatten(out, start_dim=1)
        logits = self.linear1(out)
        return logits

    def _make_res_block(self):
        layers = list()
        num_filters = self.width
        input_num_filters = num_filters
        for stack in range(3):
            for res_block in range(self.num_res_blocks):
                layers.append(
                    StackedPreResnetBlock(
                        self.normalization_layer,
                        num_filters,
                        input_num_filters,
                        stack,
                        res_block,
                        self.activation,
                        self.use_bias,
                        self.input_size,
                    )
                )
                input_num_filters = num_filters
            num_filters *= 2
        return nn.Sequential(*layers)


def make_preresnet20_frn_fn(data_info, activation=torch.nn.Identity):
    num_classes = data_info["num_classes"]
    input_size = data_info.get("input_size", 32)

    return MakePreResnetFn(
        num_classes,
        depth=20,
        normalization_layer=FilterResponseNorm,
        activation=activation,
        input_size=input_size,
    )


def make_preresnet56_frn_fn(data_info, activation=torch.nn.Identity):
    num_classes = data_info["num_classes"]
    input_size = data_info.get("input_size", 32)

    return MakePreResnetFn(
        num_classes,
        depth=56,
        normalization_layer=FilterResponseNorm,
        activation=activation,
        input_size=input_size,
    )


def make_preresnet110_frn_fn(data_info, activation=torch.nn.Identity):
    num_classes = data_info["num_classes"]
    input_size = data_info.get("input_size", 32)

    return MakePreResnetFn(
        num_classes,
        depth=110,
        normalization_layer=FilterResponseNorm,
        activation=activation,
        input_size=input_size,
    )


def make_preresnet164_frn_fn(data_info, activation=torch.nn.Identity):
    num_classes = data_info["num_classes"]
    input_size = data_info.get("input_size", 32)

    return MakePreResnetFn(
        num_classes,
        depth=164,
        normalization_layer=FilterResponseNorm,
        activation=activation,
        input_size=input_size,
    )


def make_vgg(data_info, _=None):
    net = vgg_frn.VGG(
        "VGG16", data_info["num_classes"], data_info["input_size"]
    )

    return net


def make_resnet18(data_info, _=None):
    rn = resnet_frn.ResNet18(
        num_classes=data_info["num_classes"],
        input_size=data_info["input_size"],
    )

    return rn


def make_densenet121(data_info, _=None):
    rn = densenet_frn.DenseNet121(
        num_classes=data_info["num_classes"],
        input_size=data_info["input_size"],
    )

    return rn


# pytorch version
def get_model(model_name, data_info, **kwargs):
    _MODEL_FNS = {
        "resnet20_frn": make_resnet20_frn_fn,
        "preresnet20_frn": make_preresnet20_frn_fn,
        "preresnet56_frn": make_preresnet56_frn_fn,
        "preresnet110_frn": make_preresnet110_frn_fn,
        "preresnet164_frn": make_preresnet164_frn_fn,
        "vgg16": make_vgg,
        "resnet18": make_resnet18,
        "densenet121": make_densenet121,
    }
    net_fn = _MODEL_FNS[model_name](data_info, **kwargs)
    return net_fn
