# Not All Tokens Are Equal: Human-centric Visual Analysis via Token Clustering Transformer

Our classification code is developed on top of [PVT](https://github.com/whai362/PVT).

For details please see [Not All Tokens Are Equal: Human-centric Visual Analysis via Token Clustering Transformer](https://arxiv.org/abs/2204.08680). 

If you use this code for a paper please cite:


```
@ARTICLE{2022arXiv220408680Z,
       author = {{Zeng}, Wang and {Jin}, Sheng and {Liu}, Wentao and {Qian}, Chen and {Luo}, Ping and {Ouyang}, Wanli and {Wang}, Xiaogang},
        title = "{Not All Tokens Are Equal: Human-centric Visual Analysis via Token Clustering Transformer}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition},
         year = 2022,
        month = apr,
          eid = {arXiv:2204.08680},
        pages = {arXiv:2204.08680},
archivePrefix = {arXiv},
       eprint = {2204.08680},
 primaryClass = {cs.CV},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv220408680Z},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Usage

First, clone the repository locally:
```
git clone https://github.com/zengwang430521/TCFormer.git
```
Then, install PyTorch 1.6.0+ and torchvision 0.7.0+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):

```
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
```

Then install mmcv:
```
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
```


## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Model Zoo

- TCFormer on ImageNet-1K

| Method           | Size | Acc@1 | #Params (M) | Config                                   | Checkpoint                                                                                 | log|
|------------------|:----:|:-----:|:-----------:|------------------------------------------|--------------------------------------------------------------------------------------------|----|
| TCFormer        |  224 |  82.3 |    25.6M      | [config](configs/tcformer/tcformer.py)    | 103M [[Google]](https://drive.google.com/file/d/1sIrTzIKFfW5Io2MybIWJoS0sv72Wd-av/view?usp=sharing) | [[Google]](https://drive.google.com/file/d/1xR3aMoWfU9sUznGtqRU6U9zcFCt_9MSk/view?usp=sharing)|


## Evaluation
To evaluate a pre-trained PVT-Small on ImageNet val with a single GPU run:
```
sh dist_train.sh configs/tcformer/tcformer.py 1 --data-path /path/to/imagenet --resume /path/to/checkpoint_file --eval
```
This should give
```
* Acc@1 82.346 Acc@5 95.982 loss 0.798
Accuracy of the network on the 50000 test images: 82.3%
```

## Training
To train TCFormer on ImageNet on a single node with 8 gpus for 300 epochs run:

```
sh dist_train.sh configs/tcformer/tcformer.py 8 --data-path /path/to/imagenet
```

If you can train on a cluster managed with slurm, you can use the script slurm_train.sh.
```
./slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
```

Here is an example of using 16 GPUs to train TCFormer on the dev partition in a slurm cluster.
(Use GPUS_PER_NODE=8 to specify a single slurm cluster node with 8 GPUs, CPUS_PER_TASK=2 to use 2 cpus per task. 
Assume that Test is a valid ${PARTITION} name.)
```
GPUS=16 GPUS_PER_NODE=8 CPUS_PER_TASK=2 ./slurm_train.sh Test tcformer configs/tcformer/tcformer.py work_dirs/tcformer 
```

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
