# Faster R-CNN

## 소개
Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks 논문 리뷰 및 코드 공부  
(https://arxiv.org/abs/1506.01497)  
<br>
논문 리뷰: https://velog.io/@skhim520/Faster-R-CNN-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-%EB%B0%8F-%EC%BD%94%EB%93%9C-%EA%B5%AC%ED%98%84  
참고한 코드(baseline): https://github.com/chenyuntc/simple-faster-rcnn-pytorch  


## 코드 실행
### 1. 환경설정
anaconda를 이용한 환경설정  

```sh
# create conda env
conda create --name simp python=3.7
conda activate simp
# install pytorch
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# install other dependancy
pip install visdom scikit-image tqdm fire ipdb matplotlib torchnet

# start visdom
nohup python -m visdom.server &

```

### 2. 데이터 다운 및 준비
#### 2.1 Pascal VOC2007 데이터 다운로드

```Bash
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```

#### 2.2 데이터 압축 풀기

```Bash
   tar xvf VOCtrainval_06-Nov-2007.tar
   tar xvf VOCtest_06-Nov-2007.tar
   tar xvf VOCdevkit_08-Jun-2007.tar
```

#### 2.3 데이터 폴더 경로 설정
utils/config.py 에서 voc_data_dir을 데이터가 있는 경로로 설정 또는 실행할 때 argument로 지정하기  
ex) `voc_data_dir = path/VOCdevkit/VOC2007/`

### 3. 학습하기
```Bash
python train.py train --env='fasterrcnn' --plot-every=100
```  
Some Key arguments:

- `--caffe-pretrain=False`: use pretrain model from caffe or torchvision (Default: torchvison)
- `--plot-every=n`: visualize prediction, loss etc every `n` batches.
- `--env`: visdom env for visualization
- `--voc_data_dir`: where the VOC data stored
- `--use-drop`: use dropout in RoI head, default False
- `--use-Adam`: use Adam instead of SGD, default SGD. (You need set a very low `lr` for Adam)
- `--load-path`: pretrained model path, default `None`, if it's specified, it would be loaded.

### 4. 시각화 및 학습 경과
Pytorch에서 사용할 수 있는 시각화 도구 visdom을 통해 logging을 하고 visualization을 한다.  
해당 결과를 확인하기 위해서는 다음과 같이 하면 된다.  
<br>
#### Local
웹 브라우저 url에 `http://localhost:8097`로 visdom server 접속 `8097`은 default 값이다

#### Virtual Machine
ssh로 가상 머신에 접속할 때 visdom server를 local 웹 브라우저와 연결시켜줘야 한다.

 1 `ssh <Username>@<Host> -p<port> -L localhost:18097:localhost:8097`
 
 2 웹 브라우저 url에 `http://localhost:18097`로 visdom server 접속
