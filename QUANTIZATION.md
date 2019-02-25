# QUANTIZE Guild

## 1.do quantization
```
export INT8INFO=1
python2 tools/test_net.py \     
    --cfg configs/12_2017_baselines/retinanet_R-101-FPN_1x.yaml    \ 
    --device_id -2 \
    --range 1000 2000 \  
    TEST.WEIGHTS  https://dl.fbaipublicfiles.com/detectron/36768744/12_2017_baselines/retinanet_R-101-FPN_1x.yaml.08_31_38.5poQe1ZB/output/train/coco_2014_train%3Acoco_2014_valminusminival/retinanet/model_final.pkl 
```

it will generate two pb files named as retinanet_init_int8.pb and retinanet_predict_int8.pb under the folder you run this command. 

## 2. run quantized mode
```
export INT8INFO=0
export INT8PATH='/path/to/your/detectron/'  #the detectron folder path
python2 tools/test_net.py \
    --cfg configs/12_2017_baselines/retinanet_R-101-FPN_1x.yaml \
    --device_id -2  \
    TEST.WEIGHTS  https://dl.fbaipublicfiles.com/detectron/36768744/12_2017_baselines/retinanet_R-101-FPN_1x.yaml.08_31_38.5poQe1ZB/output/train/coco_2014_train%3Acoco_2014_valminusminival/retinanet/model_final.pkl
```

