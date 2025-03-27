from collections import OrderedDict
import advertorch.utils as au
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from advertorch.attacks import LabelMixin, Attack as A
from ._differential_evolution import differential_evolution
from dml_csr.pASMA_parsing import parse

import random
import torchvision.transforms as T

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

def wrapper_method(func):
    def wrapper_func(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        for atk in self.__dict__.get('_attacks').values():
            eval("atk."+func.__name__+"(*args, **kwargs)")
        return result
    return wrapper_func

class Attack(object):
    def __init__(self, name, model):
        self.attack = name
        self._attacks = OrderedDict()

        self.set_model(model)
        self.device = next(model.parameters()).device
        self.targeted = False
        self._target_map_function = None
        # Controls attack mode.
        self.attack_mode = 'default'

        # Controls when normalization is used.
        self.normalization_used = {}
        self._normalization_applied = False

    @wrapper_method
    def _set_normalization_applied(self, flag):
        self._normalization_applied = flag

    @wrapper_method
    def set_normalization_used(self, mean, std):
        n_channels = len(mean)
        mean = torch.tensor(mean).reshape(1, n_channels, 1, 1)
        std = torch.tensor(std).reshape(1, n_channels, 1, 1)
        self.normalization_used['mean'] = mean
        self.normalization_used['std'] = std
        self._normalization_applied = True

    def normalize(self, inputs):
        mean = self.normalization_used['mean'].to(inputs.device)
        std = self.normalization_used['std'].to(inputs.device)
        return (inputs - mean) / std

    def inverse_normalize(self, inputs):
        mean = self.normalization_used['mean'].to(inputs.device)
        std = self.normalization_used['std'].to(inputs.device)
        return inputs * std + mean

    def forward(self, inputs, labels=None, *args, **kwargs):
        raise NotImplementedError

    def _check_inputs(self, images):
        tol = 1e-4
        if self._normalization_applied:
            images = self.inverse_normalize(images)
        if torch.max(images) > 1 + tol or torch.min(images) < 0 - tol:
            raise ValueError('Input must have a range [0, 1] (max: {}, min: {})'.format(
                torch.max(images), torch.min(images)))
        return images

    def _check_outputs(self, images):
        if self._normalization_applied:
            images = self.normalize(images)
        return images

    @wrapper_method
    def set_model(self, model):
        self.model = model
        self.model_name = model.__class__.__name__

    def get_logits(self, inputs, labels=None, *args, **kwargs):
        if self._normalization_applied:
            inputs = self.normalize(inputs)
        logits = self.model(inputs)
        return logits



    @wrapper_method
    def set_device(self, device):
        self.device = device


    def get_mode(self):
        return  self.attack_mode

    def __call__(self, images, labels=None, *args, **kwargs):
        images = self._check_inputs(images)
        adv_images = self.forward(images, labels, *args, **kwargs) if labels is not None \
                else self.forward(images, *args, **kwargs)
        adv_images = self._check_outputs(adv_images)
        return adv_images

    def __repr__(self):
        info = self.__dict__.copy()
        del_keys = ['model', 'attack', 'supported_mode']
        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)
        for key in del_keys:
            del info[key]
        info['attack_mode'] = self.attack_mode
        info['normalization_used'] = True if len(self.normalization_used) > 0 else False  # nopep8
        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        attacks = self.__dict__.get('_attacks')
        # Get all items in iterable items.
        def get_all_values(items, stack=[]):
            if (items not in stack):
                stack.append(items)
                if isinstance(items, list) or isinstance(items, dict):
                    if isinstance(items, dict):
                        items = (list(items.keys()) + list(items.values()))
                    for item in items:
                        yield from get_all_values(item, stack)
                else:
                    if isinstance(items, Attack):
                        yield items
            else:
                if isinstance(items, Attack):
                    yield items
        for num, value in enumerate(get_all_values(value)):
            attacks[name + "." + str(num)] = value
            for subname, subvalue in value.__dict__.get('_attacks').items():
                attacks[name + "." + subname] = subvalue




class FGSM(Attack):

    """
        FGSM in the paper 'Explaining and harnessing adversarial examples'
        [https://arxiv.org/abs/1412.6572]

        Distance Measure : Linf
    """

    def __init__(self, model, eps=8 / 255):
        super().__init__("FGSM", model)
        self.eps = eps

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss = nn.CrossEntropyLoss()
        images.requires_grad = True
        outputs = self.get_logits(images)
        # Calculate loss
        cost = loss(outputs['prob'].float(), labels.float())
        # Update adversarial images
        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]
        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        return adv_images

class PGD(Attack):

    """
        PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
        [https://arxiv.org/abs/1706.06083]

        Distance Measure : Linf
    """


    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start


    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss = nn.CrossEntropyLoss()
        #loss = self.cam_loss
        adv_images = images.clone().detach()
        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + \
                         torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)['prob']
            # Calculate loss
            cost = loss(outputs.float(), labels.float())
            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images,
                                min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()


        return adv_images

class ASMA(Attack):

    """
        PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
        [https://arxiv.org/abs/1706.06083]

        Distance Measure : Linf
    """


    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start


    def forward(self, images, labels, mask=None):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()
        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + \
                         torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)['prob']
            # Calculate loss

            cost = loss(outputs.float(), labels.float())

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()

            if mask is not None:
                adv_images = adv_images.to('cuda:0')
                mask = mask.to('cuda:0')
                adv_images = adv_images*mask
            delta = torch.clamp(adv_images - images,
                                min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

class CW(Attack):

    """
        CW in the paper 'Towards Evaluating the Robustness of Neural Networks'
        [https://arxiv.org/abs/1608.04644]

        Distance Measure : L2
    """

    def __init__(self, model, c=1, kappa=0, steps=10, lr=0.01):
        super().__init__("CW", model)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True
        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)
        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()
        optimizer = optim.Adam([w], lr=self.lr)
        for step in range(self.steps):
            # Get adversarial images
            adv_images = self.tanh_space(w)
            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()
            outputs = self.get_logits(adv_images)['prob'].float()
            # print(outputs.shape)
            f_loss = self.f(outputs, labels).sum()
            
            cost = L2_loss + self.c * f_loss
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            # Update adversarial images
            # pre = torch.argmax(outputs.detach(), 0)
            pre = (outputs > 0.5).long()
            
            condition = (pre != labels).float()
            
            # Filter out images that get either correct predictions or non-decreasing loss,
            # i.e., only images that are both misclassified and loss-decreasing are left
            mask = condition * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2
            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images
            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(self.steps // 10, 1) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()
        return best_adv_images

    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        # atanh is defined in the range -1 to 1
        return self.atanh(torch.clamp(x * 2 - 1, min=-1, max=1))

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    # f-function in the paper
   
    def f(self, outputs, labels):
        # 目标类别的 logits
        real = outputs * labels + (1 - labels) * (-outputs)  

        # 非目标类别的 logits
        other = outputs * (1 - labels) + labels * (-outputs)  
    
        # 计算 logits 差值，并截断
        return torch.clamp(real - other, min=-self.kappa)

class pASMA(Attack):
    r"""
    Attack in the paper 'One pixel attack for fooling deep neural networks'
    [https://arxiv.org/abs/1710.08864]

    Modified from "https://github.com/DebangLi/one-pixel-attack-pytorch/" and 
    "https://github.com/sarathknv/adversarial-examples-pytorch/blob/master/one_pixel_attack/"

    Distance Measure : L0

    Arguments:
        model (nn.Module): model to attack.
        pixels (int): number of pixels to change (Default: 1)
        steps (int): number of steps. (Default: 10)
        popsize (int): population size, i.e. the number of candidate agents or "parents" in differential evolution (Default: 10)
        inf_batch (int): maximum batch size during inference (Default: 128)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.OnePixel(model, pixels=1, steps=10, popsize=10, inf_batch=128)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, pixels=4, steps=50, popsize=400, inf_batch=128):
        super().__init__("OnePixel", model)
        self.pixels = pixels
        self.steps = steps
        self.popsize = popsize
        self.inf_batch = inf_batch
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        batch_size, channel, height, width = images.shape

        bounds = [(0, height), (0, width)] + [(0, 1)] * channel
        bounds = bounds * self.pixels
        bounds = []

        popmul = max(1, int(self.popsize / 4))

        adv_images = []
        for idx in range(batch_size):
            image, label = images[idx : idx + 1], labels[idx : idx + 1]

            if self.targeted:
                target_label = target_labels[idx : idx + 1]
                # print(image.shape)
                bounds = parse(image)
                

                def func(delta):
                    return self._loss(image, target_label, delta)

                def callback(delta, convergence):
                    return self._attack_success(image, target_label, delta)

            else:
                bounds = parse(image)
                #print(bounds)
                def func(delta):
                    return self._loss(image, label, delta)

                def callback(delta, convergence):
                    return self._attack_success(image, label, delta)

            delta = differential_evolution(
                func=func,
                bounds=bounds,
                callback=callback,
                maxiter=self.steps,
                popsize=popmul,
                init="random",
                recombination=1,
                atol=-1,
                polish=False,
            ).x
            delta = np.split(delta, len(delta) / len(bounds))
            adv_image = self._perturb(image, delta, size=4)
            adv_images.append(adv_image)

        adv_images = torch.cat(adv_images)
        return adv_images



    def _pen_fun(self, delta, image, lambda_penalty=1, D_max=500):
        """
        计算扰动 `delta` 在感知上的颜色变化，并返回归一化的惩罚项。

        参数：
        - delta: (N, pixels, 5) 数组，每个像素包含 (x, y, r, g, b)
        - image: 原始图像（Tensor 或 NumPy 数组）
        - lambda_penalty: 罚项系数
        - D_max: 归一化参数

        返回：
        - penaltys: (N,) 的 Tensor，表示每个 delta 的感知损失
        """
        delta = np.array(delta)
        if len(delta.shape) < 2:
            delta = np.array([delta])
        num_delta = len(delta)

        # 确保 image 是 NumPy 数组（如果是 Tensor 则转换）
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            #print(image.shape)

        penaltys = []

        for idx in range(num_delta):
            penalty = 0  # 初始化当前 delta 的累计惩罚值
            pixel_info = delta[idx].reshape(self.pixels, -1)

            for pixel in pixel_info:
                pos_x, pos_y = int(pixel[0]), int(pixel[1])  # 确保坐标是整数
                r, g, b = pixel[2], pixel[3], pixel[4]  # 读取扰动颜色

                # 获取原始像素颜色
                original_color = sRGBColor(*image[0, :, pos_y, pos_x], is_upscaled=True)
                perturbed_color = sRGBColor(r, g, b, is_upscaled=True)

                # RGB 转换为 Lab 颜色空间
                original_lab = convert_color(original_color, LabColor)
                perturbed_lab = convert_color(perturbed_color, LabColor)

                # 计算 CIEDE2000 感知颜色差异
                delta_e = delta_e_cie2000(original_lab, perturbed_lab)

                # 累加罚项
                penalty += delta_e

            # 归一化并存储
            penaltys.append(penalty * lambda_penalty / D_max)

        # 转换为 Torch Tensor 以兼容 loss 计算
        return np.array(penaltys, dtype=np.float32)
    
    def _loss(self, image, label, delta):
        adv_images = self._perturb(image, delta, size=4)  # Mutiple delta
        prob = self._get_prob(adv_images)
        penalty = self._pen_fun(delta, image)
        if label == 0:
            prob = 1 - prob
        if self.targeted:
            return 1 - prob  # If targeted, increase prob
        else:
            # print(prob.shape, penalty.shape)
            return prob  + penalty  # If non-targeted, decrease prob

    def _attack_success(self, image, label, delta):
        adv_image = self._perturb(image, delta, size=4)  # Single delta
        prob = self._get_prob(adv_image)
        pre = np.argmax(prob)
        if self.targeted and (pre == label):
            return True
        elif (not self.targeted) and (pre != label):
            return True
        return False

    def _get_prob(self, images):
        with torch.no_grad():
            batches = torch.split(images, self.inf_batch)
            outs = []
            for batch in batches:
                out = self.get_logits(batch)
                outs.append(out['prob'])
        outs = torch.cat(outs)
        prob = F.softmax(outs, dim=0)
        return prob.detach().cpu().numpy()

    def _perturb(self, image, delta, size=4):
        delta = np.array(delta)
        if len(delta.shape) < 2:
            delta = np.array([delta])
        num_delta = len(delta)
        adv_image = image.clone().detach().to(self.device)
        adv_images = torch.cat([adv_image] * num_delta, dim=0)
        for idx in range(num_delta):
            pixel_info = delta[idx].reshape(self.pixels, -1)
            for pixel in pixel_info:
                pos_x, pos_y = pixel[:2]
                pos_x, pos_y = int(pos_x), int(pos_y)
                pos_x = pos_x - size if pos_x + size > 255 else pos_x
                pos_y = pos_y - size if pos_y + size > 255 else pos_y
                channel_v = pixel[2:]
                for channel, v in enumerate(channel_v):
                    #print(v)
                    adv_images[idx, channel, pos_x:pos_x+size, pos_y:pos_y+size] = v
        return adv_images


class BSR(Attack):
    def __init__(self, model, epsilon=2/255, alpha=2/255, epoch=10, decay=1., num_scale=1, num_block=1, targeted=False, random_start=False,
                norm='linfty', loss='crossentropy', device=None, attack='BSR', **kwargs):
        super().__init__("BSR", model)
        self.attack = attack
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.epsilon = epsilon
        self.targeted = targeted
        self.random_start = random_start
        self.norm = norm
        self.num_scale = num_scale
        self.num_block = num_block
        self.loss = nn.CrossEntropyLoss() if loss == 'crossentropy' else nn.MSELoss()

    def forward(self, x, labels=None, *args, **kwargs):
        """
        The general attack procedure

        Arguments:
            data (N, C, H, W): tensor for input images
            labels (N,): tensor for ground-truth labels if untargetd
            labels (2,N): tensor for [ground-truth, targeted labels] if targeted
        """
        data = x
        if self.targeted:
            assert len(labels) == 2
            labels = labels[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):
            # Obtain the output
            adv_images = self.transform(data+delta, momentum=momentum)
            logits = self.get_logits(adv_images)['prob'].float()
            #logits = self.get_logits(self.transform(data+delta, momentum=momentum))['prob']
            
            # Calculate the loss

            loss = self.get_loss(logits, labels.float())

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta+data
    
    def init_delta(self, data, **kwargs):
        def clamp(x, x_min, x_max):
            return torch.min(torch.max(x, x_min), x_max)
        delta = torch.zeros_like(data).to(self.device)
        if self.random_start:
            if self.norm == 'linfty':
                delta.uniform_(-self.epsilon, self.epsilon)
            else:
                delta.normal_(-self.epsilon, self.epsilon)
                d_flat = delta.view(delta.size(0), -1)
                n = d_flat.norm(p=2, dim=-1).view(delta.size(0), 1, 1, 1)
                r = torch.zeros_like(data).uniform_(0,1).to(self.device)
                delta *= r/n*self.epsilon
            delta = clamp(delta, 0-data, 1-data)
        delta.requires_grad = True
        return delta

    def update_delta(self, delta, data, grad, alpha, **kwargs):
        def clamp(x, x_min, x_max):
            return torch.min(torch.max(x, x_min), x_max)
        if self.norm == 'linfty':
            delta = torch.clamp(delta + alpha * grad.sign(), -self.epsilon, self.epsilon)
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
            scaled_grad = grad / (grad_norm + 1e-20)
            delta = (delta + scaled_grad * alpha).view(delta.size(0), -1).renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(delta)
        delta = clamp(delta, 0-data, 1-data)
        return delta.detach().requires_grad_(True)

    def get_grad(self, loss, delta, **kwargs):
        """
        The gradient calculation, which should be overridden when the attack need to tune the gradient (e.g., TIM, variance tuning, enhanced momentum, etc.)
        """
        return torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
    
    def get_momentum(self, grad, momentum, **kwargs):
        """
        The momentum calculation
        """
        return momentum * self.decay + grad / (grad.abs().mean(dim=(1,2,3), keepdim=True))

    def get_length(self, length):
        rand = np.random.uniform(2, size=self.num_block)
        rand_norm = np.round(rand/rand.sum()*length).astype(np.int32)
        rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
        return tuple(rand_norm)

    def shuffle_single_dim(self, x, dim):
        lengths = self.get_length(x.size(dim))
        x_strips = list(x.split(lengths, dim=dim))
        random.shuffle(x_strips)
        return x_strips

    def image_rotation(self, x):
        rotation_transform = T.RandomRotation(degrees=(-24, 24), interpolation=T.InterpolationMode.BILINEAR)
        return  rotation_transform(x)

    def shuffle(self, x):
        dims = [2,3]
        random.shuffle(dims)
        x_strips = self.shuffle_single_dim(x, dims[0])
        return torch.cat([torch.cat(self.shuffle_single_dim(self.image_rotation(x_strip), dim=dims[1]), dim=dims[1]) for x_strip in x_strips], dim=dims[0])

    def transform(self, x, **kwargs):
        """
        Scale the input for BSR
        """
        return torch.cat([self.shuffle(x) for _ in range(self.num_scale)])

    def get_loss(self, logits, label):
        """
        Calculate the loss
        """
        return -self.loss(logits, label.repeat(self.num_scale)) if self.targeted else self.loss(logits, label)
    
class Jitter(Attack):
    r"""
    Jitter in the paper 'Exploring Misclassifications of Robust Neural Networks to Enhance Adversarial Attacks'
    [https://arxiv.org/abs/2105.10304]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.Jitter(model, eps=8/255, alpha=2/255, steps=10,
                 scale=10, std=0.1, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        model,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
        scale=10,
        std=0.1,
        random_start=True,
    ):
        super().__init__("Jitter", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.scale = scale
        self.std = std
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.MSELoss(reduction="none")

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            logits = self.get_logits(adv_images)['prob']

            #_, pre = torch.max(logits, dim=0)
            pre = (logits > 0.5).int()
            
            wrong = pre != labels
            
            norm_z = torch.norm(logits, p=float("inf"), dim=0, keepdim=True)
            hat_z = nn.Softmax(dim=0)(self.scale * logits / norm_z)


            if self.std != 0:
                hat_z = hat_z + self.std * torch.randn_like(hat_z)

            # Calculate loss
            if self.targeted:
                target_Y = F.one_hot(
                    target_labels, num_classes=logits.shape[-1]
                ).float()
                cost = -loss(hat_z, target_Y).mean(dim=1)
            else:
                Y = F.one_hot(labels, num_classes=logits.shape[-1]).float()
                cost = loss(hat_z, Y).mean(dim=1)

            norm_r = torch.norm(
                (adv_images - images), p=float("inf"), dim=[1, 2, 3]
            )  # nopep8
            nonzero_r = norm_r != 0
            cost[wrong * nonzero_r] /= norm_r[wrong * nonzero_r]

            cost = cost.mean()

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(
                adv_images - images, min=-self.eps, max=self.eps
            )  # nopep8
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
