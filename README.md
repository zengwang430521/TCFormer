# TCFormer (CVPR'2022 Oral)

\[[📜paper](https://arxiv.org/abs/2204.08680)\]

## Introduction

Official code repository for the paper:  
[**Not All Tokens Are Equal: Human-centric Visual Analysis via Token Clustering Transformer**](https://arxiv.org/abs/2204.08680)    
[Wang Zeng, Sheng Jin, Wentao Liu, Chen Qian, Ping Luo, Wanli Ouyang, and Xiaogang Wang]  


![teaser](images/fig2.png)

## TODO
- [ ] Whole-body pose estimation training/testing codes release.
- [ ] Whole-body pose estimation model zoo release.

## Model Zoo

### Image Classification

Classification configs & weights see >>>[here](classification/)<<<.

- TCFormer on ImageNet-1K

| Method           | Size | Acc@1 | #Params (M) | Config                                   | Checkpoint                                                                                 | log|
|------------------|:----:|:-----:|:-----------:|------------------------------------------|--------------------------------------------------------------------------------------------|----|
| TCFormer        |  224 |  82.3 |    25.6M      | [config](configs/tcformer/tcformer.py)    | 103M [[Google]](https://drive.google.com/file/d/1sIrTzIKFfW5Io2MybIWJoS0sv72Wd-av/view?usp=sharing) | [[Google]](https://drive.google.com/file/d/1xR3aMoWfU9sUznGtqRU6U9zcFCt_9MSk/view?usp=sharing)|


## Citation
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

## Acknowledgement

Thanks to:

- [PVT](https://github.com/whai362/PVT)
- [MMPose](https://github.com/open-mmlab/mmpose)

## License

This project is released under the [Apache 2.0 license](LICENSE).
