# VQA 모델

### 모델 선택

여러 모델을 테스트 해보면서 가장 적합한 모델을 선택 함  

* 기준
1. 인원을 정확히 찾아내는가?
2. 현재 프레임에 사람이 있는 구간과 없는 구간을 구분할 수 있는가?
3. 어떤 자세인가? 혹은 어떤 자세를 취하는지 알 수 있는가?
4. 모델의 크기가 적절한가?

* lang-chain
모델의 token을 생각을 해봤을 때 위 기준을 한번에 얻기는 어려울 것 같음  
lang-chain을 이용해서 모델을 테스트 해보고자 함  


* 모델 list

| 모델 이름     | 설명                                                                       | GitHub 링크                                                                                  | 모델 크기 |
|---------------|----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|-----------|
| **VQA²**    | 비디오 품질 평가를 위한 시각적 질의응답 모델로, VQA² 시리즈 모델과 데이터셋을 제공합니다. | [VQA² GitHub](https://github.com/Q-Future/Visual-Question-Answering-for-Video-Quality-Assessment) | 150MB     |
| **FAST-VQA**| 효율적인 비디오 품질 평가를 위한 모델로, 빠른 속도와 높은 성능을 제공합니다.       | [FAST-VQA GitHub](https://github.com/VQAssessment/FAST-VQA-and-FasterVQA)                      | 150MB     |
| **IneterVL** |                                                                            | [IneterVL](https://github.com/OpenGVLab/InternVL)                                              | 확인중    |
| **Video-LLaMA** |                                                                            | [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA)                                      | 확인중, 여러 모델이 있지만 용량이 500MB 보다 작음 |

VQA²의 경우에는 용량이 너무 커서 모델을 메모리에 올리지도 못할 것 같음    
경랑화된 모델을 기준으로 모델을 테스트를 진행 해야 될 것 같음  
