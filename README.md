<div align="center" style="line-height:1.2; margin:0; padding:0;">
<h1 style="margin-bottom:0em;">Multi-View 3D Point Tracking</h1>

<a href="https://arxiv.org/abs/2508.21060"><img src="https://img.shields.io/badge/arXiv-2508.21060-b31b1b" alt="arXiv"></a>
<a href="https://ethz-vlg.github.io/mvtracker/"><img src="https://img.shields.io/badge/Project%20Page-009688?logo=internetcomputer&logoColor=white" alt="Project Page"></a>
<a href="https://ethz-vlg.github.io/mvtracker/#qualitative-visualization"><img src="https://img.shields.io/badge/Interactive%20Results-673ab7?logo=apachespark&logoColor=white" alt="Interactive Results"></a>
[![](https://img.shields.io/badge/🤗%20Demo-Coming%20soon…-ffcc00)](#)
<br>
[**Frano Rajič**](https://m43.github.io/)<sup>1</sup> · 
[**Haofei Xu**](https://haofeixu.github.io/)<sup>1</sup> · 
[**Marko Mihajlovic**](https://markomih.github.io/)<sup>1</sup> · 
[**Siyuan Li**](https://siyuanliii.github.io/)<sup>1</sup> · 
[**Irem Demir**](https://github.com/iremddemir)<sup>1</sup>  
[**Emircan Gündoğdu**](https://github.com/emircangun)<sup>1</sup> · 
[**Lei Ke**](https://www.kelei.site/)<sup>2</sup> · 
[**Sergey Prokudin**](https://vlg.inf.ethz.ch/team/Dr-Sergey-Prokudin.html)<sup>1,3</sup> · 
[**Marc Pollefeys**](https://people.inf.ethz.ch/marc.pollefeys/)<sup>1,4</sup> · 
[**Siyu Tang**](https://vlg.inf.ethz.ch/team/Prof-Dr-Siyu-Tang.html)<sup>1</sup>
<br>
<sup>1</sup>[ETH Zürich](https://vlg.inf.ethz.ch/) &emsp;
<sup>2</sup>[Carnegie Mellon University](https://www.cmu.edu/) &emsp;
<sup>3</sup>[Balgrist University Hospital](https://www.balgrist.ch/) &emsp;
<sup>4</sup>[Microsoft](https://www.microsoft.com/)
</div>

<p float="left">
  <img alt="selfcap" src="https://github.com/user-attachments/assets/b502d193-c37c-43be-af6c-653b5de7597e" width="48%" /> 
  <img alt="dexycb" src="https://github.com/user-attachments/assets/d14d4c6c-152e-4040-b29b-3da4b7e8b913" width="48%" /> 
  <img alt="4d-dress-stretching" src="https://github.com/user-attachments/assets/f3eabdda-59e1-4032-b345-c4603ea86fc0" width="48%" />
  <img alt="4d-dress-avatarmove" src="https://github.com/user-attachments/assets/3fef9924-84ad-4295-95e2-5b82ae7c3053" width="48%" />
</p>

MVTracker is the first **data-driven multi-view 3D point tracker** for tracking arbitrary 3D points across multiple cameras. It fuses multi-view features into a unified 3D feature point cloud, within which it leverages kNN-based correlation to capture spatiotemporal relationships across views. A transformer then iteratively refines the point tracks, handling occlusions and adapting to varying camera setups without per-sequence optimization.


## Updates

- <ins>August 28, 2025</ins>: Public release.


## Quick Start

This repo was validated on **Python 3.10.12**, **PyTorch 2.3.0** (CUDA 12.1), **cuDNN 8903**, and **gcc 11.3.0**. If you want a fresh minimal environment that runs the Hub demo and `demo.py`:
```bash
conda create -n 3dpt python=3.10.12 -y
conda activate 3dpt
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r https://raw.githubusercontent.com/ethz-vlg/mvtracker/refs/heads/main/requirements.txt

# Optional, speeds up the model
pip install --upgrade --no-build-isolation flash-attn==2.5.8  # Speeds up attention
pip install "git+https://github.com/ethz-vlg/pointcept.git@2082918#subdirectory=libs/pointops"  # Speeds up kNN search; may require gcc 11.3.0: conda install -c conda-forge gcc_linux-64=11.3.0 gxx_linux-64=11.3.0 gcc=11.3.0 gxx=11.3.0
```

With the minimal dependencies in place, you can try MVTracker directly via **PyTorch Hub**:
```python
import torch
import numpy as np
from huggingface_hub import hf_hub_download

device = "cuda" if torch.cuda.is_available() else "cpu"
mvtracker = torch.hub.load("ethz-vlg/mvtracker", "mvtracker", pretrained=True, device=device)

# Example input from demo sample (downloaded automatically)
sample = np.load(hf_hub_download("ethz-vlg/mvtracker", "data_sample.npz"))
rgbs = torch.from_numpy(sample["rgbs"]).float()
depths = torch.from_numpy(sample["depths"]).float()
intrs = torch.from_numpy(sample["intrs"]).float()
extrs = torch.from_numpy(sample["extrs"]).float()
query_points = torch.from_numpy(sample["query_points"]).float()

with torch.no_grad():
    results = mvtracker(
        rgbs=rgbs[None].to(device) / 255.0,
        depths=depths[None].to(device),
        intrs=intrs[None].to(device),
        extrs=extrs[None].to(device),
        query_points_3d=query_points[None].to(device),
    )

pred_tracks = results["traj_e"].cpu()  # [T,N,3]
pred_vis = results["vis_e"].cpu()      # [T,N]
print(pred_tracks.shape, pred_vis.shape)
```

Alternatively, you can run our interactive demo:

```bash
python demo.py --rerun save --lightweight
```

By default this saves a lightweight `.rrd` recording (e.g., `mvtracker_demo.rrd`) that you can open in any Rerun viewer. The simplest option is to drag and drop the file into the [online viewer](https://app.rerun.io/version/0.21.0). For the best experience, you can also install Rerun locally (`pip install rerun-sdk==0.21.0; rerun`). Results can be explored interactively in the viewer with WASD/QE navigation, mouse rotation and zoom, and timeline playback controls.

<details>
<summary>[Interactive viewer on a cluster or with GUI support - click to expand]</summary>
  
If you are working on a cluster, you can stream results directly to your laptop by forwarding a port (`ssh -R 9876:localhost:9876 user@cluster`) and then running the demo in streaming mode (`python demo.py --rerun stream`), which sends live data into your local Rerun instance. If you are running the demo locally with GUI support, you can automatically spawn a Rerun window (`python demo.py --rerun spawn`).

</details>


## Installation

You can use a pretrained model directly via **PyTorch Hub** (see Quick Start above), or clone this repo if you want to run our demo, evaluation, or training. We recommend using **PyTorch with CUDA** for best performance. CPU-only runs are possible but very slow.

```bash
git clone https://github.com/ethz-vlg/mvtracker.git
cd mvtracker
```

To extend the conda environment from the Quick Start to support training and evaluation, install the full requirements by running `pip install -r requirements.full.txt`. Baselines based on SpatialTracker V1 also require cupy:
```bash
pip install tensorflow==2.12.1 tensorflow-datasets tensorflow-graphics tensorboard
pip install cupy-cuda12x==12.2.0
python -m cupyx.tools.install_library --cuda 12.x --library cutensor
python -m cupyx.tools.install_library --cuda 12.x --library nccl
python -m cupyx.tools.install_library --cuda 12.x --library cudnn
```


## Datasets

To benchmark multi-view 3D point tracking, we provide preprocessed versions of three datasets:

- **MV-Kubric**: a synthetic training dataset adapted from single-view Kubric into a multi-view setting.  
- **Panoptic Studio**: evaluation benchmark with real-world activities such as basketball, juggling, and toy play (10 sequences).  
- **DexYCB**: evaluation benchmark with real-world hand–object interactions (10 sequences).  

<details>
<summary>[Downloading our preprocessed datasets - click to expand]</summary>
  
You can download and extract them as (~72 GB after extraction):

```bash
# MV-Kubric (simulated + DUSt3R depths)
wget https://huggingface.co/datasets/ethz-vlg/mv3dpt-datasets/resolve/main/kubric-multiview--test.tar.gz -P datasets/
wget https://huggingface.co/datasets/ethz-vlg/mv3dpt-datasets/resolve/main/kubric-multiview--test--dust3r-depth.tar.gz -P datasets/
tar -xvzf datasets/kubric-multiview--test.tar.gz -C datasets/
tar -xvzf datasets/kubric-multiview--test--dust3r-depth.tar.gz -C datasets/
rm datasets/kubric-multiview*.tar.gz

# Panoptic Studio (optimization-based depth from Dynamic3DGS)
wget https://huggingface.co/datasets/ethz-vlg/mv3dpt-datasets/resolve/main/panoptic-multiview.tar.gz -P datasets/
tar -xvzf datasets/panoptic-multiview.tar.gz -C datasets/
rm datasets/panoptic-multiview.tar.gz

# DexYCB (Kinect + DUSt3R depths)
wget https://huggingface.co/datasets/ethz-vlg/mv3dpt-datasets/resolve/main/dex-ycb-multiview.tar.gz -P datasets/
wget https://huggingface.co/datasets/ethz-vlg/mv3dpt-datasets/resolve/main/dex-ycb-multiview--dust3r-depth.tar.gz -P datasets/
tar -xvzf datasets/dex-ycb-multiview.tar.gz -C datasets/
tar -xvzf datasets/dex-ycb-multiview--dust3r-depth.tar.gz -C datasets/
rm datasets/dex-ycb-multiview*.tar.gz

# $ du -sch datasets/*
# 31G     kubric-multiview
# 13G     panoptic-multiview
# 29G     dex-ycb-multiview
# 72G     total
```

</details>


<details>
<summary>[Regenerating datasets from scratch - click to expand]</summary>
  
If you wish to regenerate datasets from scratch, we provide scripts with docstrings that explain usage and list the commands we used. For licensing and usage terms, please refer to the original datasets. 
- MV-Kubric data for training and testing can be generated with [ethz-vlg/kubric](https://github.com/ethz-vlg/kubric/blob/multiview-point-tracking/challenges/point_tracking_3d/worker.py).
- DexYCB can be downloaded and labels regenerated using [`scripts/dex_ycb_to_neus_format.py`](./scripts/dex_ycb_to_neus_format.py); note that we have created labels for 10 sequences, but DexYCB is much larger and more labels could be produced if needed.
- Panoptic Studio can be downloaded and labels regenerated using [`scripts/panoptic_studio_preprocessing.py`](./scripts/panoptic_studio_preprocessing.py).
- DUSt3R depths can be produced for any dataset with [`scripts/estimate_depth_with_duster.py`](./scripts/estimate_depth_with_duster.py).
- For unlabeled datasets used only in qualitative experiments, we provide the following preprocessing scripts: [4D-Dress](./scripts/4ddress_preprocessing.py), [Hi4D](./scripts/hi4d_preprocessing.py), [EgoExo4D](./scripts/egoexo4d_preprocessing.py), and [SelfCap](./scripts/selfcap_preprocessing.py).  

</details>

For quick testing, we also release a small **demo sample** (~200 MB):

```bash
python demo.py --random_query_points
```

Our generic loader [`GenericSceneDataset`](./mvtracker/datasets/generic_scene_dataset.py) supports adding new datasets. It can compute depths on the fly with [DUSt3R](https://github.com/naver/dust3r), [VGGT](https://vgg-t.github.io), [MonoFusion](https://imnotprepared.github.io/research/25_DSR/index.html), or [MoGe-2](https://github.com/microsoft/MoGe), and can also estimate camera poses with VGGT.  



## Evaluation

Evaluation is driven by Hydra configs. See [`mvtracker/cli/eval.py`](./mvtracker/cli/eval.py) and [`configs/eval.yaml`](./configs/eval.yaml) for details.

To evaluate MVTracker with our best model, first download the checkpoint from [Hugging Face](https://huggingface.co/ethz-vlg/mvtracker):

```bash
wget https://huggingface.co/ethz-vlg/mvtracker/resolve/main/mvtracker_200000_june2025.pth -P checkpoints/
```

Then run:

```bash
python -m mvtracker.cli.eval \
  experiment_path=logs/mvtracker \
  model=mvtracker \
  datasets.eval.names=[kubric-multiview-v3-views0123] \
  restore_ckpt_path=checkpoints/mvtracker_200000_june2025.pth

# Expected result:
# {
#   "eval_kubric-multiview-v3-views0123/model__ate_visible__dynamic-static-mean": 5.07,
#   "eval_kubric-multiview-v3-views0123/model__average_jaccard__dynamic-static-mean": 81.42,
#   "eval_kubric-multiview-v3-views0123/model__average_pts_within_thresh__dynamic-static-mean": 90.00
# }
```

To evaluate a baseline, e.g. CoTracker3-Online (auto-downloaded checkpoint), run:

```bash
python -m mvtracker.cli.eval experiment_path=logs/cotracker3-online model=cotracker3_online

# Expected result:
# {
#   "eval_panoptic-multiview-views1_7_14_20/model__average_jaccard__any": 74.56
# }
```

For more baselines and dataset setups (e.g. varying camera counts, camera subsets, etc.), see [`scripts/slurm/eval.sh`](./scripts/slurm/eval.sh) for the commands used in our experiments.

<details>
<summary>[Details on evaluation parameters - click to expand]</summary>
  
The evaluation datasets are specified with `datasets.eval.names`. Each name is parsed by the dataset `from_name()` factory (see e.g. [`DexYCBMultiViewDataset.from_name`](./mvtracker/datasets/dexycb_multiview_dataset.py)), which supports modifiers such as `-views`, `-duster`, `-novelviews`, `-removehand`, `-2dpt`, or `-cached`. This makes it easy to select subsets of cameras, enable different depth sources, or ensure deterministic track sampling. The main labeled benchmarks are:
- **Kubric (synthetic)** — e.g. `kubric-multiview-v3-views0123`  
- **Panoptic Studio (real)** — e.g. `panoptic-multiview-views1_7_14_20`  
- **DexYCB (real)** — e.g. `dex-ycb-multiview-views0123`  

For reproducibility of our main results, we also provide *cached* variants of each benchmark, which freeze track selection exactly as used in our paper. Without `-cached`, random seeding ensures reproducibility, but cached versions guarantee identical tracks across environments. The following cached variants are included in the released datasets:
- `kubric-multiview-v3-views0123-cached`  
- `kubric-multiview-v3-duster0123-cached`  
- `panoptic-multiview-views1_7_14_20-cached`  
- `panoptic-multiview-views27_16_14_8-cached`  
- `panoptic-multiview-views1_4_7_11-cached`  
- `dex-ycb-multiview-views0123-cached`  
- `dex-ycb-multiview-duster0123-cached`  

</details>

## Training

To run a small overfitting test that fits into 24 GB GPU RAM:

```bash
python -m mvtracker.cli.train +experiment=mvtracker_overfit_mini
```

For a full-scale MVTracker on an 80 GB GPU:

```bash
python -m mvtracker.cli.train +experiment=mvtracker_overfit
```

## NuscTrack (nuScenes surround-view TAP)

Local adapter for [NuscTrack](https://github.com/Ishida-Mitsunari/mvtracker) clips under `/share/tgp/yangyi/nuscenes`: RGB + UniDepthV2, queries \((cam,t,x,y)\) lifted to ego (no GT 3D as query), 384 stratified tracks, \(432\times 768\), no scene-radius normalization.

Use the venv and keep `PYTHONPATH` pointed at this repo. Do **not** leave `CUDA_VISIBLE_DEVICES` set to a single GPU. Official Kubric weights: `checkpoints/mvtracker_200000_june2025.pth` (if `huggingface.co` hangs, download via `https://hf-mirror.com/ethz-vlg/mvtracker/resolve/main/mvtracker_200000_june2025.pth`). Install `pointops` for kNN; 6-view \(432\times 768\) training also needs encoder checkpointing (already in the model).

```bash
cd /share/tgp/yangyi/mvtracker
export PYTHONPATH=/share/tgp/yangyi/mvtracker
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Skip any busy GPU, e.g. GPU 2 occupied:
export CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7
```

Run long jobs in tmux:

```bash
tmux new -s yangyi_mvtracker          # or: tmux attach -t yangyi_mvtracker
```

**CoTracker3 eval** (clip-sharded processes, not DDP; protocol metrics in the log dir):

```bash
# Smoke (2 clips)
.venv/bin/python -m mvtracker.cli.eval_nusctrack_parallel \
  --dataset nusctrack-val-max2 --gpus all \
  --log-dir /share/tgp/yangyi/mvtracker/logs/cotracker3_offline_nusctrack

# Full val
.venv/bin/python -m mvtracker.cli.eval_nusctrack_parallel \
  --dataset nusctrack-val --gpus all \
  --log-dir /share/tgp/yangyi/mvtracker/logs/cotracker3_offline_nusctrack_val
```

Hydra equivalents: `configs/eval_nusctrack_cotracker3.yaml` (max2) and `configs/eval_nusctrack_cotracker3_val.yaml` (full val).

**MVTracker Kubric-pretrained fine-tune** on `nusctrack-train` (AdamW \(5\times 10^{-5}\), 7000 steps, same experiment dir auto-resumes):

```bash
.venv/bin/python -m mvtracker.cli.train +experiment=mvtracker_nusctrack_ft
# or: bash scripts/nusctrack_ft.sh
```

In-training eval uses `nusctrack-val-max2`. After training, full val:

```bash
.venv/bin/python -m mvtracker.cli.eval \
  +experiment=mvtracker_nusctrack_ft \
  modes.eval_only=true \
  datasets.eval.names=[nusctrack-val] \
  restore_ckpt_path=/share/tgp/yangyi/mvtracker/logs/mvtracker_nusctrack_ft/model_final.pth \
  experiment_path=/share/tgp/yangyi/mvtracker/logs/mvtracker_nusctrack_ft_eval
```

## Practical Considerations

<details>
<summary>[Scene normalization - click to expand]</summary>

Performance depends strongly on scene normalization. MVTracker was trained on Kubric with randomized but bounded scales and camera setups. At test time, scenes with very different scales, rotations, or translations must be aligned to this distribution. Our generic loader provides an automatic normalization that assumes the ground plane is parallel to the XY plane. This automatic normalization worked reasonably well for 4D-Dress, Hi4D, EgoExo4D, and SelfCap. For Panoptic and DexYCB, we applied manual similarity transforms, which are encoded in the respective dataloaders. Robust, general-purpose normalization remains an open challenge.  

</details>


<details>
<summary>[Challenges and future directions - click to expand]</summary>

The central challenge in multi-view 3D point tracking is 4D reconstruction: obtaining depth maps that are accurate, temporally consistent, and available in real time, especially under sparse-view setups. MVTracker performs well when sensor depth and camera calibration are provided, but in settings where both must be estimated, errors in reconstruction quickly make tracking unreliable. While learned motion priors help tolerate moderate noise, they cannot replace a robust reconstruction backbone. We believe progress will hinge on methods that jointly solve depth estimation and tracking for mutual refinement, or large-scale foundation models for 4D reconstruction and tracking that fully leverage data and compute. We hope the community will direct future efforts toward this goal.


</details>


## Acknowledgements

Our code builds upon and was inspired by many prior works, including [SpaTracker](https://github.com/henry123-boy/SpaTracker), [CoTracker](https://github.com/facebookresearch/co-tracker), and [DUSt3R](https://github.com/naver/dust3r). We thank the authors for releasing their code and pretrained models. We are also grateful to maintainers of [Rerun](https://rerun.io) for their helpful visualization toolkit.

## Citation

If you find our repository useful, please consider giving it a star ⭐ and citing our work:
```bibtex
@inproceedings{rajic2025mvtracker,
  title     = {Multi-View 3D Point Tracking},
  author    = {Raji{\v{c}}, Frano and Xu, Haofei and Mihajlovic, Marko and Li, Siyuan and Demir, Irem and G{\"u}ndo{\u{g}}du, Emircan and Ke, Lei and Prokudin, Sergey and Pollefeys, Marc and Tang, Siyu},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```
