# chainer-Grad-CAM
Chainer版 Grad-CAM [元の論文](https://arxiv.org/abs/1611.07450).
## 準備
numpy, opencv, matplotlib, chainer

chainer用のVGGモデル
## 実行
```
python grad_cam.py path_to_image
```
--labelオプションでターゲットカテゴリを指定できる。デフォルトは0(トップ1のカテゴリ)

## 結果
```
python grad_cam.py images/cat_dog.png --label 0
```

| 出力 |
| ------|
| ![output image][out]|

[out]: https://github.com/ninhydrin/chainer-Grad-CAM/blob/master/result/cat_dog.png



