# trafficsign
## 赛题
[旷视AI智慧交通开源赛道](https://studio.brainpp.com/competition/4?name=%E6%97%B7%E8%A7%86AI%E6%99%BA%E6%85%A7%E4%BA%A4%E9%80%9A%E5%BC%80%E6%BA%90%E8%B5%9B%E9%81%93&tab=overview)，初赛1/177，复赛1/11。    
本赛题为复杂场景的交通标志检测，对五种交通标志进行识别。
## 框架
megengine
## 算法方案
- 网络框架
  - atss + resnext101_32x8d
    
- 训练阶段
    
  - **图片尺寸**  
    最终提交版本输入图片尺寸为(1500,2100)
      
  - **多尺度训练**（最终提交版本未采用）  
    起初我们将短边设为(1024, 1056, 1088, 1120, 1152, 1184, 1216, 1248, 1280, 1312, 1344, 1376, 1408)，随机选取短边后，长边按比例缩放，并使长边长度小于1800，从而进行多尺度训练，这取得了很好的效果。
    不过后期的mosaic和mixup在增强时对图片进行了缩放，实则隐含了多尺度训练，且效果优于上述方法，我们最终去掉了多尺度训练。  
      
  - **数据增强**
    - **mosaic增强**
      
      随机组合四张图片，随机平移10%，尺度缩放(0.5,2.0)，shear 0.1。  
        
    - **mixup增强**
      
      随机选取两张图进行叠加，我们最终选用的比例是0.5*原图+0.5*新图片，同时其进行缩放(0.5,2.0)。  
      
    - **随机水平翻转**
      
      直接对图片进行翻转，会导致第三个类别“arr_l”（左转线）和右转线混淆，故我们添加了class-aware的翻转，遇到有“arr_l”类的图片则不进行翻转。  
        
    - **基于Albumentations库的各种增强**（最终提交版本未采用）
      
      我们尝试了ShiftScaleRotate（验证集+0.5）、CLANE（验证集+1.0）、RandomBrightnessContrast等，但组合起来测试集提点欠佳，所以最后没用。  
        
    - **gridmask增强**（最终提交版本未采用）
      
      提点欠佳，所以没采用。  
        
    - **类别平衡采样**（最终提交版本未采用）
        
      使用类别平衡采样后，效果不是很好，这可能是因为数据集本身没有严重的类别不均衡。下面是我们统计的每个类别在图片中出现的频率。
      | |红灯|直行线|左转线|禁止行驶|禁止停车|
      |---|---|---|---|---|---|
      |频率|0.356|0.228|0.201|0.257|0.485|
    
 - **多尺度测试**
   - **多尺度测试图片尺寸**
   
       最后提交版本(2100,2700),(2100,2800),(2400,3200)，如果继续增加尺度，map还会继续提高。
      
   - **topk—nms**
      
      对上述三个尺度生成的结果先进行nms，再将得到的结果框与剩下所有框进行topk—nms（保留与当前结果框iou大于0.85的top3的框，把这些框的坐标进行融合），参数设置vote_thresh=0.85, k=3。
    
   - **固定最短边的多尺度测试**（最终提交版本未采用）
   
      效果没上面的好，所以最后没采用。
      
 - **网络结构** 
   
    - backbone从res50到res101再到resx101有稳定涨点。
    
    - 我们还在backbone部分尝试了dcn和gcnet，收效甚微，最终没有采用。
    
## 模型训练与测试

 - **数据集位置** 
```
/path/to/...    
    |->traffic...    
    |    |images...     
    |    |annotations->|train.json...     
    |    |             |val.json...     
    |    |             |test.json...      
```
 - **训练测试**
   
 在加上增强后，我们训练了36个epoch。
```
pip3 install --user -r requirements.txt
export PYTHONPATH=your_path/trafficsign:$PYTHONPATH
cd weights；wget https://data.megengine.org.cn/models/weights/atss_resx101_coco_2x_800size_45dot6_b3a91b36.pkl
python3 tools/train.py -n 4 -b 2 -f configs/atss_resx101_final.py -d your_datasetpath -w weights/atss_resx101_coco_2x_800size_45dot6_b3a91b36.pkl
python3 tools/test_final.py -n 4 -se 35 -f configs/atss_resx101_final.py -d your_datasetpath 
```
  (-n 能抢到几张卡就写几吧qaq)


