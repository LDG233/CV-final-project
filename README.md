# Handwritten Mathematical Expression Recognition

This is the final project of AI3604, 2023-2024 Fall. 

### Environment

```
python==3.8.5
numpy==1.22.2
opencv-python==4.5.5.62
PyYAML==6.0
tensorboardX==2.5
torch==1.6.0+cu101
torchvision==0.7.0+cu101
tqdm==4.64.0
```

### Train

```
python train.py --config path_to_train_config_yaml
```

### Dataset

CROHME

```
python data.py --config path_to_data_config_yaml
```

### Reference

Bohan Li, Ye Yuan and etc. When Counting Meets HMER: Counting-Aware Network for 
Handwritten Mathematical Expression Recognition. In Proceedings of the European 
Conference on Computer Vision (ECCV), 2022.
Ye Yuan, Xiao Liu, Wondimu Dikubab and etc. Syntax-Aware Network for Handwritten 
Mathematical Expression Recognition. In Proceedings of IEEE Conference on Computer 
Vision and Pattern Recognition (CVPR), 2022