import torch
import numpy as np
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import os
import pickle as pkl
from pathlib import Path
import tempfile
import shutil
# from mmseg.core import mean_iou
from PIL import Image
from skimage import measure
# from skimage.measure import compare_ssim, compare_psnr
# from skimage.measure import structural_similarity as compare_ssim
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import lpips
from torchvision.transforms import transforms
from scipy import linalg
from torchvision.models.inception import inception_v3
from torch.autograd import Variable
from scipy.stats import entropy
import torch.utils.data
from skimage import color, io


"""
ImageNet classifcation accuracy
"""


def accuracy(output, target, topk=(1,)):
    """
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)      # pred: return indices.(the class of the max value.)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            correct_k /= batch_size
            res.append(correct_k)
        return res


def classify_acc(q_pred, q_actual, topk=(1, )):
    # print('q_pred.size', q_pred.size(), q_actual.size())    # [b, 313, h, w]
    batch_size, n_cls, h, w = q_pred.shape[0], q_pred.shape[1], q_pred.shape[2], q_pred.shape[3]
    with torch.no_grad():
        maxk = max(topk)
        _, pred = q_pred.detach().topk(maxk, 1, True, True)      # [b, 5, h, w]
        pred = pred.permute(0, 2, 3, 1).contiguous().view(batch_size * h * w, maxk)      # [b * h *w, 5]
        # print(pred.device, q_pred.device)
        pred_label = torch.zeros(batch_size * h * w, n_cls).to(pred.device).scatter(1, pred, 5)

        _, actual = q_actual.detach().topk(maxk, 1, True, True)  # [b, 5, h, w]
        actual = actual.permute(0, 2, 3, 1).contiguous().view(batch_size * h * w, maxk)
        actual_label = torch.ones(batch_size * h * w, n_cls).to(actual.device).scatter(1, actual, 5)

        correct = pred_label.eq(actual_label)
        correct_num = torch.sum(correct)
        acc = correct_num / (batch_size * maxk * h * w)
        return acc.item()


"""
Segmentation mean IoU
based on collect_results_cpu
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/apis/test.py#L160-L200
"""


def gather_data(seg_pred, tmp_dir=None):
    """
    distributed data gathering
    prediction and ground truth are stored in a common tmp directory
    and loaded on the master node to compute metrics
    """
    if tmp_dir is None:
        tmpprefix = os.path.expandvars("$WORK/temp")
    else:
        tmpprefix = tmp_dir
    MAX_LEN = 512
    # 32 is whitespace
    dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8, device=ptu.device)
    if ptu.dist_rank == 0:
        tmpdir = tempfile.mkdtemp(prefix=tmpprefix)
        tmpdir = torch.tensor(
            bytearray(tmpdir.encode()), dtype=torch.uint8, device=ptu.device
        )
        dir_tensor[: len(tmpdir)] = tmpdir
    # broadcast tmpdir from 0 to to the other nodes
    dist.broadcast(dir_tensor, 0)
    tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    tmpdir = Path(tmpdir)
    """
    Save results in temp file and load them on main process
    """
    tmp_file = tmpdir / f"part_{ptu.dist_rank}.pkl"
    pkl.dump(seg_pred, open(tmp_file, "wb"))
    dist.barrier()
    seg_pred = {}
    if ptu.dist_rank == 0:
        for i in range(ptu.world_size):
            part_seg_pred = pkl.load(open(tmpdir / f"part_{i}.pkl", "rb"))
            seg_pred.update(part_seg_pred)
        shutil.rmtree(tmpdir)
    return seg_pred


def compute_metrics(
    seg_pred,
    seg_gt,
    n_cls,
    ignore_index=None,
    ret_cat_iou=False,
    tmp_dir=None,
    distributed=False,
):
    ret_metrics_mean = torch.zeros(3, dtype=float, device=ptu.device)
    if ptu.dist_rank == 0:
        list_seg_pred = []
        list_seg_gt = []
        keys = sorted(seg_pred.keys())
        for k in keys:
            list_seg_pred.append(np.asarray(seg_pred[k]))
            list_seg_gt.append(np.asarray(seg_gt[k]))
        ret_metrics = mean_iou(
            results=list_seg_pred,
            gt_seg_maps=list_seg_gt,
            num_classes=n_cls,
            ignore_index=ignore_index,
        )
        ret_metrics = [ret_metrics["aAcc"], ret_metrics["Acc"], ret_metrics["IoU"]]
        ret_metrics_mean = torch.tensor(
            [
                np.round(np.nanmean(ret_metric.astype(np.float)) * 100, 2)
                for ret_metric in ret_metrics
            ],
            dtype=float,
            device=ptu.device,
        )
        cat_iou = ret_metrics[2]
    # broadcast metrics from 0 to all nodes
    if distributed:
        dist.broadcast(ret_metrics_mean, 0)
    pix_acc, mean_acc, miou = ret_metrics_mean
    ret = dict(pixel_accuracy=pix_acc, mean_accuracy=mean_acc, mean_iou=miou)
    if ret_cat_iou and ptu.dist_rank == 0:
        ret["cat_iou"] = cat_iou
    return ret

####################################################################################
###################### Evalutaion Model #############################
def lab_to_rgb(img):
    assert img.dtype == np.float32
    return (255 * np.clip(color.lab2rgb(img), 0, 1)).astype(np.uint8)

def rgb_to_lab(img):
    assert img.dtype == np.uint8
    return color.rgb2lab(img).astype(np.float32)


class INCEPTION_V3(nn.Module):
    def __init__(self, incep_state_dict):
        super(INCEPTION_V3, self).__init__()
        self.model = models.inception_v3()
        self.model.load_state_dict(incep_state_dict)
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, input):
        # [-1.0, 1.0] --> [0, 1.0]
        x = input * 0.5 + 0.5
        # mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        # --> mean = 0, std = 1
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225
        #
        # --> fixed-size input: batch x 3 x 299 x 299
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
        # 299 x 299 x 3
        x = self.model(x)
        x = nn.Softmax(dim=-1)(x)
        return x

class INCEPTION_V3_FID(nn.Module):
    """pretrained InceptionV3 network returning feature maps"""
    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,  # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 incep_state_dict,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True):
        """Build pretrained InceptionV3
        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, normalizes the input to the statistics the pretrained
            Inception network expects
        """
        super(INCEPTION_V3_FID, self).__init__()

        self.resize_input = resize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = models.inception_v3()
        inception.load_state_dict(incep_state_dict)
        for param in inception.parameters():
            param.requires_grad = False

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear')

        x = x.clone()
        # [-1.0, 1.0] --> [0, 1.0]
        x = x * 0.5 + 0.5
        x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp

def get_activations(images, model, batch_size, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : the images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size depends
                     on the hardware.
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    #d0 = images.shape[0]
    d0 = int(images.size(0))
    if batch_size > d0:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = d0

    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, 2048))
    for i in range(n_batches):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches), end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        '''batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
        batch = Variable(batch, volatile=True)
        if cfg.CUDA:
            batch = batch.cuda()'''
        batch = images[start:end]

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print(' done')

    return pred_arr


def calculate_activation_statistics(act):
    """Calculation of the statistics used by the FID.
    Params:
    -- act      : Numpy array of dimension (n_images, dim (e.g. 2048)).
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_psnr_np(img1, img2):
    import numpy as np
    SE_map = (1.*img1-img2)**2
    cur_MSE = np.mean(SE_map)
    return 20*np.log10(255./np.sqrt(cur_MSE))


def avg_ssim_psnr(predicted_dir, gt_dir):
    img_names = os.listdir(gt_dir)
    img_counts = 0
    total_ssim, total_psnr = 0, 0
    total_ssim_convert, total_psnr_convert = 0, 0
    for img in img_names:
        print('calculate ssim/psnr', img_counts)
        gt_pth = os.path.join(gt_dir, img)
        gt = np.array(Image.open(gt_pth).convert('RGB')).astype(np.uint8)
        # need gt: rgb-> lab-> rgb
        gt_lab = rgb_to_lab(gt)
        gt_rgb = lab_to_rgb(gt_lab)

        pred_pth = os.path.join(predicted_dir, 'fake_' + img)
        predicted = np.array(Image.open(pred_pth).convert("RGB"))
        ssim = measure.compare_ssim(gt, predicted, data_range=225, multichannel=True)
        ssim_convert = measure.compare_ssim(gt_rgb, predicted, data_range=225, multichannel=True)
        psnr = measure.compare_psnr(gt, predicted, 255)
        psnr_convert = measure.compare_psnr(gt_rgb, predicted, 255)
        total_ssim += ssim
        total_psnr += psnr
        total_ssim_convert += ssim_convert
        total_psnr_convert += psnr_convert
        img_counts += 1
    assert img_counts == 5000
    ssim_avg = total_ssim / img_counts
    psnr_avg = total_psnr / img_counts
    ssim_avg_convert = total_ssim_convert / img_counts
    psnr_avg_convert = total_psnr_convert / img_counts
    return ssim_avg, psnr_avg, ssim_avg_convert, psnr_avg_convert


def avg_lpips(predicted_dir, gt_dir):
    img_names = os.listdir(gt_dir)
    img_counts = 0
    total_lpips, total_lpips_convert = 0, 0
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
    with torch.no_grad():
        for img in img_names:
            print('calculate lpips', img_counts)
            gt_pth = os.path.join(gt_dir, img)
            gt = np.array(Image.open(gt_pth).convert("RGB")).astype(np.uint8)
            # gt: convert rgb->lab->rgb
            gt_lab = rgb_to_lab(gt)
            gt_rgb = lab_to_rgb(gt_lab)

            gt = transforms.ToTensor()(gt)
            gt = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(gt).cuda()
            gt_rgb = transforms.ToTensor()(gt_rgb)
            gt_rgb = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(gt_rgb).cuda()

            pred_pth = os.path.join(predicted_dir, "fake_" + img)

            pred = np.array(Image.open(pred_pth).convert("RGB"))
            pred = transforms.ToTensor()(pred)
            pred = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(pred).cuda()

            lpips_vgg = loss_fn_vgg(gt, pred).cpu()
            lpips_vgg_convert = loss_fn_vgg(gt_rgb, pred).cpu()
            total_lpips += lpips_vgg
            total_lpips_convert += lpips_vgg_convert
            img_counts += 1
    assert img_counts == 5000
    lpips_avg = total_lpips / img_counts
    lpips_avg_convert = total_lpips_convert / img_counts
    return lpips_avg, lpips_avg_convert


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))
    with torch.no_grad():
        for i, batch in enumerate(dataloader, 0):
            # print('i', i)
            batch = batch[0]        # batch[0]=pred_img, batch[1]=gt_img.
            batch = batch.type(dtype)
            batchv = Variable(batch)
            batch_size_i = batch.size()[0]

            preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


class vip_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, gt_dir):
        # self.gt_dir = '/userhome/SUN_text2img/ImageNet/val'
        self.gt_dir = gt_dir
        self.data_dir = data_dir
        pre_names = os.listdir(self.gt_dir)
        self.filenames = pre_names
        self.transform_list = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        img_pth = self.filenames[index]
        img_dir = os.path.join(self.gt_dir, img_pth)
        img = Image.open(img_dir).convert('RGB')

        # convert gt.
        img_np = np.array(img).astype(np.uint8)
        img_lab = rgb_to_lab(img_np)
        img_rgb = lab_to_rgb(img_lab)

        gt_img = self.transform_list(img).float()
        gt_img_convert = self.transform_list(img_rgb).float()

        fake_dir = os.path.join(self.data_dir, 'fake_'+img_pth)

        fake_img = Image.open(fake_dir).convert("RGB")
        fake_img = self.transform_list(fake_img).float()

        return fake_img, gt_img, gt_img_convert

    def __len__(self):
        return len(self.filenames)


def calculate_is(pred_dir):
    is_mean, is_std = inception_score(vip_dataset(pred_dir), cuda=True, batch_size=32, resize=True, splits=10)
    return is_mean, is_std


def calculate_fid(pred_dir, gt_dir):
    batch_size = 1
    new_batch_size = 1
    incep_pth = os.path.join('resources', "inception_v3_google-1a9a5a14.pth")

    incep_state_dict = torch.load(incep_pth, map_location=lambda storage, loc: storage)
    block_idx = INCEPTION_V3_FID.BLOCK_INDEX_BY_DIM[2048]
    inception_model_fid = INCEPTION_V3_FID(incep_state_dict, [block_idx])
    inception_model_fid.cuda()
    inception_model_fid.eval()
    dataset = vip_dataset(pred_dir, gt_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    fake_acts_set, acts_set, acts_set_convert = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(dataloader, 0):
            pred, gt, gt_convert = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
            fake_act = get_activations(pred, inception_model_fid, new_batch_size)
            real_act = get_activations(gt, inception_model_fid, new_batch_size)
            real_act_convert = get_activations(gt_convert, inception_model_fid, new_batch_size)
            fake_acts_set.append(fake_act)
            acts_set.append(real_act)
            acts_set_convert.append(real_act_convert)
            # break
        acts_set = np.concatenate(acts_set, 0)
        fake_acts_set = np.concatenate(fake_acts_set, 0)
        acts_set_convert = np.concatenate(acts_set_convert, 0)

        real_mu, real_sigma = calculate_activation_statistics(acts_set)
        fake_mu, fake_sigma = calculate_activation_statistics(fake_acts_set)
        real_mu_convert, real_sigma_convert = calculate_activation_statistics(acts_set_convert)
        fid_score = calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)
        fid_score_convert = calculate_frechet_distance(real_mu_convert, real_sigma_convert, fake_mu, fake_sigma)
    return fid_score, fid_score_convert