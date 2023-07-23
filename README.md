[PyTorch] DINO: self-DIstillation with NO labels
=====
PyTorch implementation of "Emerging Properties in Self-Supervised Vision Transformers"

## Concept
<div align="center">
  <img src="./figures/dino.png" width="400">    
  <p>Concept ot the DINO [1].</p>
</div>

## Results
||Student ($x_1$)|Teacher ($x_2$)|
|:---|:---:|:---:|
|Before|<img src="https://github.com/YeongHyeon/DINO-Toy_Pytorch/blob/main/figures/epoch_000000_s.png?raw=true" width="350">  |<img src="https://github.com/YeongHyeon/DINO-Toy_Pytorch/blob/main/figures/epoch_000000_t.png?raw=true" width="350">|  
|After|<img src="https://github.com/YeongHyeon/DINO-Toy_Pytorch/blob/main/figures/epoch_000129_s.png?raw=true" width="350">  |<img src="https://github.com/YeongHyeon/DINO-Toy_Pytorch/blob/main/figures/epoch_000129_t.png?raw=true" width="350">|  

## Requirements
* PyTorch 2.0.1

## Reference
[1] Mathilde Caron, et al. <a href="https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html">"Emerging Properties in Self-Supervised Vision Transformers."</a> Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.