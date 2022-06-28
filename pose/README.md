# Not All Tokens Are Equal: Human-centric Visual Analysis via Token Clustering Transformer

Our classification code is developed on top of [mmpose](https://github.com/open-mmlab/mmpose).

For details please see [Not All Tokens Are Equal: Human-centric Visual Analysis via Token Clustering Transformer](https://arxiv.org/abs/2204.08680). 

If you use this code for a paper please cite:


```
@inproceedings{zeng2022not,
  title={Not All Tokens Are Equal: Human-centric Visual Analysis via Token Clustering Transformer},
  author={Zeng, Wang and Jin, Sheng and Liu, Wentao and Qian, Chen and Luo, Ping and Ouyang, Wanli and Wang, Xiaogang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11101--11111},
  year={2022}
}
```


## Usage

Install [mmpose](https://github.com/open-mmlab/mmpose).

or

```
pip install mmpose
```


## Data preparation

Prepare COCO-Wholebody according to the guidelines in [mmpose](https://github.com/open-mmlab/mmpose/blob/master/docs/en/tasks/2d_wholebody_keypoint.md).


## Results and models

-  on COCO-Wholebody

Results on COCO-WholeBody v1.0 val with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | Body AP | Body AR | Foot AP | Foot AR | Face AP | Face AR  | Hand AP | Hand AR | Whole AP | Whole AR | ckpt | log |
| :---- | :--------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-----: | :-----: | :------: |:-------: |:------: | :------: |
| [TCFormer](/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/tcformer_mta_coco_wholebody_256x192.py)  | 256x192 | 0.697 | 0.774 | 0.705 | 0.821 | 0.656 | 0.753 | 0.539 | 0.652 | 0.576 | 0.681 | [ckpt](https://drive.google.com/file/d/1tRMhOxiab8BcuImi7B64BRgXPZ2A3BAx/view?usp=sharing) | [log](https://drive.google.com/file/d/1chRPtfEOPJzcuCZ7-nGsPZXZO9kjXdOH/view?usp=sharing) |
| [TCFormer_large](/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/tcformer_large_mta_coco_wholebody_384x288.py)  | 384x288 | 0.718 | 0.794 | 0.744 | 0.850 | 0.790 | 0.856 | 0.614 | 0.715 | 0.642 | 0.733 | [ckpt](https://drive.google.com/file/d/1aUIj_-U1EfklVGzELUrierwFNoUp-zrH/view?usp=sharing) | [log](https://drive.google.com/file/d/1p1TTbTg09o4mJf4vDUCrFWPhsPxg-7j7/view?usp=sharing) |




## Evaluation
You can follow the guideline of [mmpose](https://github.com/open-mmlab/mmpose/blob/master/docs/en/get_started.md)

Assume that you have already downloaded the checkpoints to the directory ```checkpoints/```.

1. Test TCFormer on COCO-WholeBody and evaluate the mAP.

   ```shell
   ./tools/dist_test.sh configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/tcformer_mta_coco_wholebody_256x192.py \
       checkpoints/tcformer_mta_256x192-68d5f8aa_20220606.pth 1 \
       --eval mAP
   ```


2. Test TCFormer on COCO-WholeBody with 8 GPUS and evaluate the mAP.

   ```shell
   ./tools/dist_test.sh configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/tcformer_mta_coco_wholebody_256x192.py \
       checkpoints/tcformer_mta_256x192-68d5f8aa_20220606.pth 8 \
       --eval mAP
   ```


3. Test TCFormer on COCO-WholeBody in slurm environment and evaluate the mAP.

   ```shell
   ./tools/slurm_test.sh mm_human test_job \
       configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/tcformer_mta_coco_wholebody_256x192.py \
       checkpoints/tcformer_mta_256x192-68d5f8aa_20220606.pth \
       --eval mAP
   ```


## Training
You can follow the guideline of [mmpose](https://github.com/open-mmlab/mmpose/blob/master/docs/en/get_started.md)


### Train with multiple GPUs
```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

To train TCFormer on COCO-WholeBody with 8 GPUS:
```shell
./tools/dist_train.sh configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/tcformer_mta_coco_wholebody_256x192.py \
  8 --work-dir work_dirs/wholebody/tcformer_mta_256
```

### Train with multiple machines

If you can run MMPose on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `slurm_train.sh`. (This script also supports single machine training.)

```shell
./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
```

Here is an example of using 16 GPUs to train TCFormer on the dev partition in a slurm cluster.
(Use `GPUS_PER_NODE=8` to specify a single slurm cluster node with 8 GPUs, `CPUS_PER_TASK=2` to use 2 cpus per task.
Assume that `Test` is a valid ${PARTITION} name.)

```shell
GPUS=16 GPUS_PER_NODE=8 CPUS_PER_TASK=2 ./tools/slurm_train.sh Test tcformer \
  configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/tcformer_mta_coco_wholebody_256x192.py \
  work_dirs/wholebody/tcformer_mta_256
```
