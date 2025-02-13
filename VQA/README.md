# VQA 모델

### 모델 선택

여러 모델을 테스트 해보면서 가장 적합한 모델을 선택 함  

* 기준
1. 인원을 정확히 찾아내는가?
2. 현재 프레임에 사람이 있는 구간과 없는 구간을 구분할 수 있는가?
3. 어떤 자세인가? 혹은 어떤 자세를 취하는지 알 수 있는가?

* lang-chain
모델의 token을 생각을 해봤을 때 위 기준을 한번에 얻기는 어려울 것 같음  
lang-chain을 이용해서 모델을 테스트 해보고자 함  


* 모델 list

| 모델 이름           | 설명                                                                 | GitHub 링크                                                                 |
|--------------------|----------------------------------------------------------------------|-----------------------------------------------------------------------------|
| **VQA²**           | 비디오 품질 평가를 위한 시각적 질의응답 모델로, VQA² 시리즈 모델과 데이터셋을 제공합니다. | [VQA² GitHub](https://github.com/Q-Future/Visual-Question-Answering-for-Video-Quality-Assessment) |
| **FAST-VQA**       | 효율적인 비디오 품질 평가를 위한 모델로, 빠른 속도와 높은 성능을 제공합니다. | [FAST-VQA GitHub](https://github.com/VQAssessment/FAST-VQA-and-FasterVQA) |
| **FasterVQA**      | FAST-VQA의 개선된 버전으로, 3D 프래그먼트 샘플링을 통해 4배 더 빠른 속도를 구현하였습니다. | [FasterVQA GitHub](https://github.com/VQAssessment/FAST-VQA-and-FasterVQA) |
| **VideoQA**        | 비디오 질의응답을 위한 다양한 코드, 데이터셋, 논문 등을 모아놓은 리포지토리입니다. | [VideoQA GitHub](https://github.com/VRU-NExT/VideoQA) |
| **Video-Question-Answering_Resources** | 비디오 질의응답에 대한 논문, 모델, 데이터셋 등의 자료를 정리한 가이드입니다. | [Video-Question-Answering_Resources GitHub](https://github.com/chakravarthi589/Video-Question-Answering_Resources) |

