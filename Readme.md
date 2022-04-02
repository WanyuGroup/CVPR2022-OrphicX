
# OrphicX
## This repo covers the implementation for our paper OrphicX.
Wanyu Lin, Hao Lan, Hao Wang and Baochun Li. "[OrphicX: A Causality-Inspired Latent Variable Model for Interpreting Graph Neural Networks](https://arxiv.org/pdf/2203.15209.pdf)," in the Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022), Oral Presentation, New Orleans, Louisiana, June 19-24, 2022.
 
## Prepare Python environment
```
conda create -n orphicx python=3.8.8
conda activate orphicx
conda install pytorch cudatoolkit=10.2 -c pytorch
conda install opencv scikit-learn networkx pandas tqdm matplotlib seaborn
pip install tensorboardx
pip install torch-scatter     -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-sparse      -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-cluster     -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-geometric
```

## Extract dataset and checkpoint from zip file
```sh
unzip ckpt.zip
```

### Run experiments with checkpoints

```sh
python orphicx_node.py --gpu --dataset syn1 --plot_info_flow
python orphicx_node.py --gpu --dataset syn4 --plot_info_flow
python orphicx_graph.py --gpu --dataset Mutagenicity --plot_info_flow
python orphicx_graph.py --gpu --dataset NCI1 --plot_info_flow
```

Calcualte information flow is slow on CPU. If you don't have a GPU, please run following commands:
```sh
python orphicx_node.py --dataset syn1
python orphicx_node.py --dataset syn4
python orphicx_graph.py --dataset Mutagenicity
python orphicx_graph.py --dataset NCI1
```

### Retrain OrphicX from scratch
```sh
python orphicx_node.py --gpu --dataset syn1 --output syn1_retrain --retrain
python orphicx_node.py --gpu --dataset syn4 --output syn4_retrain --retrain
python orphicx_graph.py --gpu --dataset Mutagenicity --output mutag_retrain --retrain
python orphicx_graph.py --gpu --dataset NCI1 --output nci1_retrain --retrain
```
