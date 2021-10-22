# Text Detection Pytorch

## 1. Dataset

### 1.1 Word Level - Scene Text

```bash
https://drive.google.com/file/d/1R9K8LZqPWM2PSNUJhy4dOnlVuQAsacYz/view?usp=sharing
```

|Dataset Name|Train|Valid|Test|Label Format|Image Extent|
|:----------:|:---:|:---:|:--:|:----------:|:----------:|
|totaltext|1255|0|80|.json|.jpg, .JPG|
|icdar2011|410|0|0|.json|.jpg, .png|
|focused_scene_text_2013|229|0|233|.json|.jpg|
|incidental_scene_text_2015|1000|0|500|.json|.jpg|

### 1.2 Text Line Level

|Document Type|Dataset Name|Train|Valid|Test|Label Format|
|:-----------:|:----------:|:---:|:---:|:--:|:----------:|
|ADMINISTRATIVE DOCUMENT|nms_vbhc_239_final_vofffice_splitword|120|37|30|json|
|ADMINISTRATIVE DOCUMENT|nms_vbhc_t7_1746_rechecked|1334|332|494|json|
|ADMINISTRATIVE DOCUMENT|nms_vbhc_t8_4301_rechecked|2719|850|679|json|
|ADMINISTRATIVE DOCUMENT|nms_vbhc_t8_4301_rechecked|2317|724|579|json|
|BUSINESS LICENSE|BUSINESS_LICENSE|556|174|138|json|
