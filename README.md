# CNN_ImageClassification
제가 Keras의 이미 학습된 VGG모델을 먼저 실행보고나서 제 CNN을 만들었습니다
다음과 같이 3가지의 사물을 학습한 결과입니다.
https://www.youtube.com/watch?v=BgfvWWw0GbE


pip install keras
pip install opencv-python
#실행방법
먼저 학습데이터를 만들기 위해 webimage.py를 실행해서 directory와 사믈의 이름 및 몇개를 찍을 수를 줘서 비디오를 찍읍니다
### 예: 꾝 data/train와 data/valid라는 파일 생성하셔야 합니다
python webimage.py data/train/사믈이름1 2000 </br>
python webimage.py data/valid/사믈이름1 200 </br>

학습한 결과를 확인하는 valid 데이터셋도 만듭니다</br>
python webimage.py data/train/사믈이름2 2000</br>
python webimage.py data/valid/사믈이름2 200</br>

학습한 결과를 확인하는 valid 데이터셋도 만듭니다

데이터셋을 만든 다음 python model2.py을 실행해서 모델을 학습하기 시작합니다
학습이 끝나고 나서 python predictme.py을 실행하며 결과를 알 수 있습니다.
