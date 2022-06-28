# TCFormer (CVPR'2022 Oral)

\[[📜paper](https://arxiv.org/abs/2204.08680)\]

## Introduction

Official code repository for the paper:  
[**Not All Tokens Are Equal: Human-centric Visual Analysis via Token Clustering Transformer**](https://arxiv.org/abs/2204.08680)    
[Wang Zeng, Sheng Jin, Wentao Liu, Chen Qian, Ping Luo, Wanli Ouyang, and Xiaogang Wang]  


![teaser](images/fig2.png)

## TODO
- [x] Whole-body pose estimation training/testing codes release.
- [x] Whole-body pose estimation model zoo release.
- [x] TCFormer-large on COCO-WholeBody dataset.
- [x] Flops calculation function.

## Model Zoo
You can find the pretrained checkpoints [here](https://drive.google.com/drive/folders/1HUzEuQnWG-LvyhAz96CthYVZdaSpllTu?usp=sharing).

### Image Classification

Classification configs & weights see >>>[here](classification/)<<<.

- TCFormer on ImageNet-1K

| Method           | Size | Acc@1 | #Params (M) | Config                                          | Checkpoint                                                                                 | log|
|------------------|:----:|:-----:|:-----------:|-------------------------------------------------|--------------------------------------------------------------------------------------------|----|
| TCFormer-light   |  224 |  79.4 |    14.2M    | [config](configs/tcformer/tcformer_light.py)    | 57M [[Google]](https://drive.google.com/file/d/1TvcJCQhHaxXJhGo13i6b5PWErhzIivuD/view?usp=sharing) | [[Google]](https://drive.google.com/file/d/11mb8v_Afx0oDEAD24pYYJSAOEgyo6g2_/view?usp=sharing)|
| TCFormer         |  224 |  82.3 |    25.6M    | [config](configs/tcformer/tcformer.py)          | 103M [[Google]](https://drive.google.com/file/d/1sIrTzIKFfW5Io2MybIWJoS0sv72Wd-av/view?usp=sharing) | [[Google]](https://drive.google.com/file/d/1xR3aMoWfU9sUznGtqRU6U9zcFCt_9MSk/view?usp=sharing)|
| TCFormer-large   |  224 |  83.6 |    62.8M    | [config](configs/tcformer/tcformer_large.py)    | 103M [[Google]](https://drive.google.com/file/d/1wu9joQJU807IGW51mIlhK4dNnMze8E1K/view?usp=sharing) | [[Google]](https://drive.google.com/file/d/1iLMSHa4YqnUdtJYeHFqVFLEJOwHwNQBN/view?usp=sharing)|


### WholeBody Estimation

WholeBody Estimation configs & weights see >>>[here](pose/)<<<.

- Results on COCO-WholeBody v1.0 val with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | Body AP | Body AR | Foot AP | Foot AR | Face AP | Face AR  | Hand AP | Hand AR | Whole AP | Whole AR | ckpt | log |
| :---- | :--------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-----: | :-----: | :------: |:-------: |:------: | :------: |
| [TCFormer](/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/tcformer_mtagather_noextra_coco_wholebody_256x192.py)  | 256x192 | 0.697 | 0.774 | 0.705 | 0.821 | 0.656 | 0.753 | 0.539 | 0.652 | 0.576 | 0.681 | [ckpt](https://drive.google.com/file/d/1tRMhOxiab8BcuImi7B64BRgXPZ2A3BAx/view?usp=sharing) | [log](https://drive.google.com/file/d/1chRPtfEOPJzcuCZ7-nGsPZXZO9kjXdOH/view?usp=sharing) |
| [TCFormer_large](/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/tcformer_large_mta_coco_wholebody_384x288.py)  | 384x288 | 0.718 | 0.794 | 0.744 | 0.850 | 0.790 | 0.856 | 0.614 | 0.715 | 0.642 | 0.733 | [ckpt](https://drive.google.com/file/d/1aUIj_-U1EfklVGzELUrierwFNoUp-zrH/view?usp=sharing) | [log](https://drive.google.com/file/d/1p1TTbTg09o4mJf4vDUCrFWPhsPxg-7j7/view?usp=sharing) |


## Citation
If you use this code for a paper, please cite:

```
@inproceedings{zeng2022not,
  title={Not All Tokens Are Equal: Human-centric Visual Analysis via Token Clustering Transformer},
  author={Zeng, Wang and Jin, Sheng and Liu, Wentao and Qian, Chen and Luo, Ping and Ouyang, Wanli and Wang, Xiaogang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11101--11111},
  year={2022}
}
```

## Acknowledgement

Thanks to:

- [PVT](https://github.com/whai362/PVT)
- [MMPose](https://github.com/open-mmlab/mmpose)

## License

This project is released under the [Apache 2.0 license](LICENSE).
