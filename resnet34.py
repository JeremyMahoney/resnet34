#%%
import json
import os
import sys
from collections import OrderedDict
from io import BytesIO
from typing import Optional, Union
import requests
import torch as t
import torchvision
from einops import rearrange
from IPython.display import display
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn
from torch.nn.functional import conv1d as torch_conv1d
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm.auto import tqdm
sys.path.append("..")
import utils
import math

MAIN = __name__ == "__main__"

URLS = [
    "https://www.oregonzoo.org/sites/default/files/styles/article-full/public/animals/H_chimpanzee%20Jackson.jpg",
    "https://anipassion.com/ow_userfiles/plugins/animal/breed_image_56efffab3e169.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/f/f2/Platypus.jpg",
    "https://static5.depositphotos.com/1017950/406/i/600/depositphotos_4061551-stock-photo-hourglass.jpg",
    "https://img.nealis.fr/ptv/img/p/g/1465/1464424.jpg",
    "http://www.tudobembresil.com/wp-content/uploads/2015/11/nouvelancopacabana.jpg",
    "https://ychef.files.bbci.co.uk/976x549/p0639ffn.jpg",
    "https://www.thoughtco.com/thmb/Dk3bE4x1qKqrF6LBf2qzZM__LXE=/1333x1000/smart/filters:no_upscale()/iguana2-b554e81fc1834989a715b69d1eb18695.jpg",
    "https://i.redd.it/mbc00vg3kdr61.jpg",
    "https://static.wikia.nocookie.net/disneyfanon/images/a/af/Goofy_pulling_his_ears.jpg",
]

#%%
def load_image(url: str) -> Image.Image:
    """Return the image at the specified URL, using a local cache if possible.

    Note that a robust implementation would need to take more care cleaning the image names.
    """
    os.makedirs("./images", exist_ok=True)
    filename = os.path.join("./images", url.rsplit("/", 1)[1].replace("%20", ""))
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            data = f.read()
    else:
        response = requests.get(url)
        data = response.content
        with open(filename, "wb") as f:
            f.write(data)
    return Image.open(BytesIO(data))


if MAIN:
    images = [load_image(url) for url in tqdm(URLS)]
    display(images[0])
    plt.imshow(images[0])

#%%

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) # why in the world these numbers?
])

def prepare_data(images: list[Image.Image]) -> t.Tensor:
    """Preprocess each image and stack them into a single tensor.

    Return: shape (batch=len(images), num_channels=3, height=224, width=224)
    """
    processedData = t.stack([preprocess(image) for image in tqdm(images)], dim=0)
    return processedData

preparedData = prepare_data(images)

with open("imagenet_labels.json") as f:
    imagenet_labels = list(json.load(f).values())

def predict(model, images: list[Image.Image], print_topk_preds=3) -> list[int]:
    """
    Pass the images through the model and print out the top predictions.

    For each image, `display()` the image and the most likely categories according to the model.

    Return: for each image, the index of the top prediction.
    """
    model.eval()
    batch = prepare_data(images)
    prediction = model(batch)
    class_ids = prediction.argmax(dim=1)
    class_names = list(imagenet_labels[i] for i in class_ids)
    for i in range(len(class_ids)):
        display(images[i])
        print(class_names[i])
    return prediction


if MAIN:
    model = models.resnet34(weights="DEFAULT")
    pretrained_categories = predict(model, images)

# %%
def einsum_trace(a: t.Tensor):
    """Compute the trace of the square 2D input using einsum."""
    return t.einsum('ii->', a)

def as_strided_trace(a: t.Tensor):
    """Compute the trace of the square 2D input using as_strided and sum.

    Tip: the stride argument to `as_strided` must account for the stride of the inputs `a.stride()`.
    """
    a_length = a.shape[0]
    return t.as_strided(a, (a_length,), (a.stride()[0]+a.stride()[1],)).sum().item()

# %%
def einsum_matmul(a: t.Tensor, b: t.Tensor) -> t.Tensor:
    """Matrix multiply 2D matrices a and b (same as a @ b)."""
    return t.einsum('ij,jk->ik', a, b)

def as_strided_matmul(a: t.Tensor, b: t.Tensor) -> t.Tensor:
    """Matrix multiply 2D matrices a and b (same as a @ b).

    Use elementwise multiplication and sum.

    Tip: the stride argument to `as_strided` must account for the stride of the inputs `a.stride()` and `b.stride()`.
    """
    assert (a.shape[1] == b.shape[0])
    return (t.as_strided(a, (a.shape[0], a.shape[1], b.shape[1]), (a.stride()[0], a.stride()[1], 0)) *
            t.as_strided(b, (a.shape[0], b.shape[0], b.shape[1]), (0, b.stride()[0], b.stride()[1]))).sum(dim=1)

# %%
def conv1d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    """Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    """
    batch, in_channels, width = x.shape
    out_channels, in_channels, kernel_width = weights.shape
    out_width = width - kernel_width + 1

    vec = t.as_strided(x, (batch, in_channels, out_width, kernel_width), (x.stride()[0], x.stride()[1], x.stride()[2], x.stride()[2]))
    out = t.einsum('biwu,oiu->bow', vec, weights)

    return out

#%%
def conv2d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    """Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    """
    batch, in_channels, height, width = x.shape
    out_channels, in_channels, kernel_height, kernel_width = weights.shape
    output_height = height - kernel_height + 1
    output_width = width - kernel_width + 1
    vec = t.as_strided(x,
    (batch, in_channels, output_height, output_width, kernel_height, kernel_width),
    (x.stride()[0], x.stride()[1], x.stride()[2], x.stride()[3], x.stride()[2], x.stride()[3]))
    out = t.einsum('bihwrc,oirc->bohw', vec, weights)
    return out

# %%
def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    """Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    """
    batch, in_channels, width = x.shape
    padding_left = t.ones(batch, in_channels, left)*pad_value
    padding_right = t.ones(batch, in_channels, right)*pad_value
    padded = t.cat((padding_left, x, padding_right), -1)
    return padded

def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
    """Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    """
    batch, in_channels, height, width = x.shape
    padding_left = t.ones(batch, in_channels, height, left)*pad_value
    padding_right = t.ones(batch, in_channels, height, right)*pad_value
    padding_top = t.ones(batch, in_channels, top, left+right+width)*pad_value
    padding_bottom = t.ones(batch, in_channels, bottom, left+right+width)*pad_value
    padded_leftright = t.cat((padding_left, x, padding_right), -1)
    padded = t.cat((padding_top, padded_leftright, padding_bottom), -2)
    return padded

# %%
def conv1d(x, weights, stride: int = 1, padding: int = 0) -> t.Tensor:
    """Like torch's conv1d using bias=False.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    """
    import math

    padded = pad1d(x, left=padding, right=padding, pad_value=0)
    batch, in_channels, width = padded.shape
    out_channels, in_channels_from_weights, kernel_width = weights.shape
    out_width = math.floor((width-kernel_width)/stride + 1)

    assert in_channels == in_channels_from_weights, "out_channels of inputs and weights should be equal"

    vec = t.as_strided(padded,
                      (batch, in_channels, out_width, kernel_width),
                      (padded.stride()[0], padded.stride()[1], stride, padded.stride()[2]))
    return t.einsum('biwu,oiu->bow', vec, weights)

# %%
IntOrPair = Union[int, tuple[int, int]]
Pair = tuple[int, int]

def force_pair(v: IntOrPair) -> Pair:
    """Convert v to a pair of int, if it isn't already."""
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)
# %%
def conv2d(x, weights, stride: IntOrPair = 1, padding: IntOrPair = 0):
    """Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    """
    import math
    stride = force_pair(stride)
    padding = force_pair(padding)
    padded = pad2d(x, padding[1], padding[1], padding[0], padding[0], 0)
    batch, in_channels, height, width = padded.shape
    out_channels, in_channels_from_weights, kernel_height, kernel_width = weights.shape
    out_width = math.floor((width-kernel_width)/stride[1]+1)
    out_height = math.floor((height-kernel_height)/stride[0]+1)

    assert in_channels == in_channels_from_weights, "in_channels from input and weights should be equal"

    vec = t.as_strided(padded,
                      (batch, in_channels, out_height, out_width, kernel_height, kernel_width),
                      (padded.stride()[0], padded.stride()[1], stride[0]*padded.stride()[2], stride[1]*padded.stride()[3], padded.stride()[2], padded.stride()[3]))
    return t.einsum('bihwrc,oirc->bohw', vec, weights)

# %%
def maxpool2d(
    x: t.Tensor, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 0
) -> t.Tensor:
    """Like torch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, height, width)
    """
    # The documented return shape isn't right
    import math
    kernel_size = force_pair(kernel_size)
    stride = force_pair(stride) if stride != None else kernel_size
    padding = force_pair(padding)
    padded = pad2d(x, padding[1], padding[1], padding[0], padding[0], float('-inf'))
    batch, channels, height, width = padded.shape
    out_height = math.floor((height-kernel_size[0])/stride[0]+1)
    out_width = math.floor((width-kernel_size[1])/stride[1]+1)
    vec = t.as_strided(padded,
                      (batch, channels, out_height, out_width, kernel_size[0], kernel_size[1]),
                      (padded.stride()[0], padded.stride()[1], padded.stride()[2]*stride[0], padded.stride()[3]*stride[1], padded.stride()[2], padded.stride()[3]))
    return vec.amax(dim=(4, 5))

# %%
class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 1):
        super().__init__()
        self.kernel_size = force_pair(kernel_size)
        self.stride = force_pair(stride) if stride != None else kernel_size
        self.padding = force_pair(padding)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Call the functional version of maxpool2d."""
        return maxpool2d(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        """Add additional information to the string representation of this class."""
        return "MaxPool2d with kernel size of {kernel_size}, stride of {stride}, and padding of {padding}".format(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

# %%
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        interval_max = 1.0/math.sqrt(in_features)
        self.weight = nn.Parameter(2*interval_max*t.rand(out_features, in_features)-interval_max)
        self.bias = nn.Parameter(2*interval_max*t.rand(out_features)-interval_max) if bias==True else None

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        return t.matmul(self.weight, x.T).T+t.as_strided(self.bias, (self.weight.shape[0], x.shape[0]), (self.bias.stride()[0], 0)).T

    def extra_repr(self) -> str:
        return "Linear module with {in_features} input features, {out_features} output features, and bias={bias}".format(in_features=self.in_features, out_features=self.out_features, bias=self.bias)

# %%
class Conv2d(t.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0
    ):
        """Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = force_pair(kernel_size)
        self.stride = force_pair(stride)
        self.padding = force_pair(padding)
        self.in_features = in_channels*math.prod(self.kernel_size)
        self.sample_limit = (1.0/(math.sqrt(self.in_features)))
        self.weight = nn.Parameter(2*self.sample_limit*t.rand(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])-self.sample_limit)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Apply the functional conv2d."""
        return conv2d(x, self.weight, self.stride, self.padding)

    def extra_repr(self) -> str:
        """"""
        return "{in_channels} input channels, {out_channels} output channels, a kernel size of {kernel_size}, a stride of {stride}, {padding_r} rows of padding, and {padding_c} columns of padding".format(
            in_channels = self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding_r=self.padding[0], padding_c=self.padding[1]
        )

#%%
class BatchNorm2d(nn.Module):
    running_mean: t.Tensor
    "running_mean: shape (num_features,)"
    running_var: t.Tensor
    "running_var: shape (num_features,)"
    num_batches_tracked: t.Tensor
    "num_batches_tracked: shape ()"

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        """Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        """
        super().__init__()
        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0))
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        """
        if self.training:
            var = t.var(x, dim=(0, 2, 3), unbiased=False)
            mean = t.mean(x, dim=(0, 2, 3))
            normalized_x = ((x-mean[None, :, None, None])/(t.sqrt(var[None, :, None, None])+self.eps))*self.weight[None, :, None, None]+self.bias[None, :, None, None]
            self.num_batches_tracked += 1
            self.running_mean = (1-self.momentum) * self.running_mean + (self.momentum) * mean
            self.running_var = (1-self.momentum) * self.running_var + (self.momentum) * var
        else:
            normalized_x = ((x-self.running_mean[None, :, None, None])/(t.sqrt(self.running_var[None, :, None, None])+self.eps))*self.weight[None, :, None, None]+self.bias[None, :, None, None]
        return normalized_x

    def extra_repr(self) -> str:
        return "{num_features} features, {eps} eps, and {momentum} momentum".format(num_features=self.num_features, eps=self.eps, momentum=self.momentum)

# %%
class ReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.maximum(t.zeros(x.shape), x)
# %%
class Sequential(nn.Module):
    def __init__(self, *modules: nn.Module):
        """
        Call `self.add_module` on each provided module, giving each one a unique (within this Sequential) name.
        Internally, this adds them to the dictionary `self._modules` in the base class, which means they'll be included in self.parameters() as desired.
        """
        super().__init__()
        for i in range(len(modules)):
            self.add_module("{module_type}_{module_number}".format(module_type=type(modules[i]).__name__, module_number=i), modules[i])

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Chain each module together, with the output from one feeding into the next one."""
        output = x
        for module in self.children():
            output = module(output)
        return output

# %%
class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        """Flatten out dimensions from start_dim to end_dim, inclusive of both.

        Return a view if possible, otherwise a copy.
        """
        flattened_dims_prod = t.prod(t.tensor(input.shape[self.start_dim:self.end_dim]))*input.shape[self.end_dim]
        first_dims = t.tensor(input.shape[:self.start_dim])
        last_dims = t.tensor(input.shape[(self.end_dim+1):]) if self.end_dim != -1 else t.tensor([])
        new_shape = t.cat((first_dims, t.tensor([flattened_dims_prod]), last_dims))
        return t.reshape(input, tuple(new_shape.int().tolist()))

    def extra_repr(self) -> str:
        return "Flatten module starting at dimension {start_dim} and ending at dimension {end_dim}, inclusive.".format(start_dim=self.start_dim, end_dim=self.end_dim)

# %%
class AveragePool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        """
        return t.mean(x, dim=(-1, -2))

#%%
class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        """A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        """
        super().__init__()
        self.left_branch = Sequential(
            Conv2d(in_feats, out_feats, kernel_size=3, stride=first_stride, padding=1),
            BatchNorm2d(out_feats),
            ReLU(),
            Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_feats)
        )
        self.right_branch = (
            Sequential (
                Conv2d(in_feats, out_feats, kernel_size=(1, 1), stride=first_stride),
                BatchNorm2d(out_feats)
            )
            if first_stride != 1
            else None
        )
        self.relu = ReLU()


    def forward(self, x: t.Tensor) -> t.Tensor:
        """Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)

        If no downsampling block is present, the addition should just add the left branch's output to the input.
        """
        left_output = self.left_branch(x)
        right_output = x if self.right_branch is None else self.right_branch(x)
        output = self.relu(left_output+right_output)
        return output
#%%
class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        """An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride."""
        super().__init__()
        self.blocks = Sequential(
            ResidualBlock(in_feats, out_feats, first_stride),
            *(
                ResidualBlock(out_feats, out_feats)
                for i in range(1, n_blocks)
            )
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Compute the forward pass.
        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        """
        return self.blocks(x)
#%%
class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()

        first_conv = Conv2d(3, out_features_per_group[0], 7, 2, 3)
        bn = BatchNorm2d(out_features_per_group[0])
        relu = ReLU()
        max_pool = MaxPool2d(3, 2, 1) # why does the solution have a stride of 2 here?

        in_features_per_group = [out_features_per_group[0]] + out_features_per_group[:-1]
        params_list = zip(n_blocks_per_group, in_features_per_group, out_features_per_group, strides_per_group)
        block_groups_list = []
        for params in params_list:
            block_groups_list.append(BlockGroup(*params))

        average_pool = AveragePool()
        flatten = Flatten() # Not sure these next two are right. Why is flatten useful here?
        linear = Linear(out_features_per_group[-1], n_classes)

        layers = [first_conv, bn, relu, max_pool] + block_groups_list + [average_pool, flatten, linear]
        self.model = Sequential(*layers)


    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, channels, height, width)

        Return: shape (batch, n_classes)
        """
        return self.model(x)

#%%

if MAIN:
    """
    Loading pre-trained weights from torchvision's resnet34 model into my model
    """
    model = models.resnet34(weights="ResNet34_Weights.DEFAULT")
    pretrained_values = model.state_dict().values()
    myModel = ResNet34()
    myModel_keys = myModel.state_dict().keys()
    new_state_dict = dict(zip(myModel_keys, pretrained_values))
    myModel.load_state_dict(new_state_dict)


# %%
def check_nan_hook(module: nn.Module, inputs, output):
    """Example of a hook function that can be registered to a module."""
    x = inputs[0]
    if t.isnan(x).any():
        raise ValueError(module, x)
    out = output[0]
    if t.isnan(out).any():
        raise ValueError(module, x)


def add_hook(module: nn.Module) -> None:
    """Remove any existing hooks and register our hook.

    Use model.apply(add_hook) to recursively apply the hook to model and all submodules.
    """
    utils.remove_hooks(module)
    module.register_forward_hook(check_nan_hook)

if MAIN:
    """
    Running my model and viewing its predictions
    """
    myModel.apply(add_hook)
    myModel_predictions = predict(myModel, images)
    
# %%
cifar_classes = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


def get_cifar10():
    """Download (if necessary) and return the CIFAR10 dataset."""
    "The following is a workaround for this bug: https://github.com/pytorch/vision/issues/5039"
    if sys.platform == "win32":
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context
    "Magic constants taken from: https://docs.ffcv.io/ffcv_examples/cifar10.html"
    mean = t.tensor([125.307, 122.961, 113.8575]) / 255
    std = t.tensor([51.5865, 50.847, 51.255]) / 255
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    cifar_train = torchvision.datasets.CIFAR10("cifar10_train", transform=transform, download=True, train=True)
    cifar_test = torchvision.datasets.CIFAR10("cifar10_train", transform=transform, download=True, train=False)
    return (cifar_train, cifar_test)


if MAIN:
    (cifar_train, cifar_test) = get_cifar10()
    trainloader = DataLoader(cifar_train, batch_size=512, shuffle=True, pin_memory=True)
    testloader = DataLoader(cifar_test, batch_size=512, pin_memory=True)
if MAIN:
    batch = next(iter(trainloader))
    print("Mean value of each channel: ", batch[0].mean((0, 2, 3)))
    print("Std value of each channel: ", batch[0].std((0, 2, 3)))
    (fig, axes) = plt.subplots(ncols=5, figsize=(15, 5))
    for (i, ax) in enumerate(axes):
        ax.imshow(rearrange(batch[0][i], "c h w -> h w c"))
        ax.set(xlabel=cifar_classes[batch[1][i].item()])
# %%
MODEL_FILENAME = "./resnet34_cifar10.pt"
device = "cuda" if t.cuda.is_available() else "cpu"


def train(trainloader: DataLoader, epochs: int) -> ResNet34:
    model = ResNet34(n_classes=10).to(device).train()
    optimizer = t.optim.Adam(model.parameters())
    loss_fn = t.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for (i, (x, y)) in enumerate(tqdm(trainloader)):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, train loss is {loss}")
        print(f"Saving model to: {os.path.abspath(MODEL_FILENAME)}")
        t.save(model, MODEL_FILENAME)
    return model


if MAIN:
    if os.path.exists(MODEL_FILENAME):
        print("Loading model from disk: ", MODEL_FILENAME)
        model = t.load(MODEL_FILENAME)
    else:
        print("Training model from scratch")
        model = train(trainloader, epochs=8)
# %%
if MAIN:
    model.eval()
    model.apply(add_hook)
    loss_fn = t.nn.CrossEntropyLoss(reduction="sum")
    with t.inference_mode():
        n_correct = 0
        n_total = 0
        loss_total = 0.0
        for (i, (x, y)) in enumerate(tqdm(testloader)):
            x = x.to(device)
            y = y.to(device)
            with t.autocast(device):
                y_hat = model(x)
                loss_total += loss_fn(y_hat, y).item()
            n_correct += (y_hat.argmax(dim=-1) == y).sum().item()
            n_total += len(x)
    print(f"Test accuracy: {n_correct} / {n_total} = {100 * n_correct / n_total:.2f}%")
    print(f"Test loss: {loss_total / n_total}")
# %%
