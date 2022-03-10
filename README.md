

# <div align="center"> **L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained Environments (Pytorch)** </div>


<p align="center">
  <img src="https://github.com/Ahmednull/storage/blob/main/gaze.gif" alt="animated" width=650 />
</p>


## Paper details

### Authors
Ahmed A.Abdelrahman, Thorsten Hempel, Aly Khalifa, Ayoub Al-Hamadi, ICIP, 2022, "under review". [[Arxiv]](https://arxiv.org/abs/2203.03339)


### Abstract
> Human gaze is a crucial cue used in various applications such as human-robot interaction and virtual reality. Recently, convolution neural network (CNN) approaches have made notable progress in predicting gaze direction. However, estimating gaze in-the-wild is still a challenging problem due to the uniqueness of eye appearance, lightning conditions, and the diversity of head pose and gaze directions. In this paper, we propose a robust CNN-based model for predicting gaze in unconstrained settings. We propose to regress each gaze angle separately to improve the per-angel prediction accuracy, which will enhance the overall gaze performance. In addition, we use two identical losses, one for each angle, to improve network learning and increase its generalization. We evaluate our model with two popular datasets collected with unconstrained settings. Our proposed model achieves state-of-the-art accuracy of 3.92 and 10.41 on MPIIGaze and Gaze360 datasets, respectively.

### Citation
If you use any part of our code or data, please cite our paper.
```
@misc{AAbdelrahman2022L2CSNetFG,
      title={L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained Environments}, 
      author={Ahmed A.Abdelrahman, Thorsten Hempel, Aly Khalifa, Ayoub Al-Hamadi},
      year={2022},
      eprint={2203.03339},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
### Evaluation

<div align="center"> 
  
<table>
<tr><th>Evaluation on MPIIGaze dataset</th><th>Evaluation on Gaze360 dataset</th></tr>
<tr><td>

|           **Methods**           | **MPIIFaceGaze** |
|:-------------------------------:|:----------------:|
|        iTracker (AlexNet)       |        5.6       |
|              MeNets             |        4.9       |
| FullFace |        4.8       |
|           Dilated-Net           |        4.8       |
|         RT-Gene(1 model)        |        4.8       |
|             GEDDNet             |        4.5       |
|       RT-Gene(4 ensemble)       |        4.3       |
|             FAR-Net             |        4.3       |
|        Bayesian Approach        |        4.3       |
|              CA-Net             |        4.1       |
|             AGE-Net             |       4.09       |
|             **L2CS-Net** **(<img src="https://render.githubusercontent.com/render/math?math=\beta"> =2)**        |   **3.96**       |
|             **L2CS-Net** **(<img src="https://render.githubusercontent.com/render/math?math=\beta"> =1)**        |   **3.92**   | 

  
</td><td>
  
 |      **Methods**     | **Front 180** | **Front Facing** |
|:--------------------:|:-------------:|:----------------:|
|       FullFace       |     14.99     |        N/A       |
|      Dilated-Net     |     13.73     |        N/A       |
| RT-Gene (4 ensemble) |     12.26     |        N/A       |
|        CA-Net        |     12.26     |        N/A       |
|    Gaze360 (LSTM)    |      11.4     |       11.1       |
|  **L2CS-Net**  **(<img src="https://render.githubusercontent.com/render/math?math=\beta"> =2)**  |     **10.54**     |       **9.13**       |
|  **L2CS-Net**  **(<img src="https://render.githubusercontent.com/render/math?math=\beta"> =1)** |     **10.41**     |       **9.01**       |

  
 </td></tr> </table>
  
</div>
 
___

## Installation
<img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" />

* Set up a virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```
* Install required packages:
```
pip install -r requirements.txt  
```

## Demo
* Install the face detector:
```sh
pip install git+https://github.com/elliottzheng/face-detection.git@master
```
* Download the pre-trained models from [here](https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd?usp=sharing) and Store it to *models/*.
*  Run:
```
 python demo.py \
 --snapshot models/L2CSNet_gaze360.pkl \
 --gpu 0 \
 --cam 0 \
```
This means the demo will run using *L2CSNet_gaze360.pkl* pretrained model


## MPIIGaze
We provide the code for train and test MPIIGaze dataset with leave-one-person-out evaluation.

### Prepare datasets
* Download **MPIIFaceGaze dataset** from [here](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation).
* Apply data preprocessing from [here](http://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/).
* Store the dataset to *datasets/MPIIFaceGaze*.

### Train
```
 python train.py \
 --dataset mpiigaze \
 --snapshot output/snapshots \
 --gpu 0 \
 --num_epochs 50 \
 --batch_size 16 \
 --lr 0.00001 \
 --alpha 1 \

```
This means the code will perform leave-one-person-out training automatically and store the models to *output/snapshots*.

### Test
```
 python test.py \
 --dataset mpiigaze \
 --snapshot output/snapshots/snapshot_folder \
 --evalpath evaluation/L2CS-mpiigaze  \
 --gpu 0 \
```
This means the code will perform leave-one-person-out testing automatically and store the results to *evaluation/L2CS-mpiigaze*.

To get the average leave-one-person-out accuracy use:
```
 python leave_one_out_eval.py \
 --evalpath evaluation/L2CS-mpiigaze  \
 --respath evaluation/L2CS-mpiigaze  \
```
This means the code will take the evaluation path and outputs the leave-one-out gaze accuracy to the *evaluation/L2CS-mpiigaze*.

## Gaze360
We provide the code for train and test Gaze360 dataset with train-val-test evaluation.

### Prepare datasets
* Download **Gaze360 dataset** from [here](http://gaze360.csail.mit.edu/download.php).

* Apply data preprocessing from [here](http://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/).

* Store the dataset to *datasets/Gaze360*.


### Train
```
 python train.py \
 --dataset gaze360 \
 --snapshot output/snapshots \
 --gpu 0 \
 --num_epochs 50 \
 --batch_size 16 \
 --lr 0.00001 \
 --alpha 1 \

```
This means the code will perform training and store the models to *output/snapshots*.

### Test
```
 python test.py \
 --dataset gaze360 \
 --snapshot output/snapshots/snapshot_folder \
 --evalpath evaluation/L2CS-gaze360  \
 --gpu 0 \
```
This means the code will perform testing on snapshot_folder and store the results to *evaluation/L2CS-gaze360*.

