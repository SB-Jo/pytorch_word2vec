# Word2Vec을 Pytorch로 구현

## utils.py
- yield_tokens
  - torchtext를 이용한 vocab(단어사전) 만들기 위한 함수
  - torchtext.vocab의 build_vocab_from_iterator에 입력으로 넣기 위해 iterator 반환
 
- create_contexts_target
  - word2vec에 사용될 (타겟-문맥 데이터 생성)
    - I like coffee에서 like가 타겟 단어인 경우, [like, [I, coffee]] 형태로 데이터셋 구성
  - 네거티브 샘플링 적용
    - Word2Vec 논문에서 주장한 방법대로 네거티브 샘플링 적용 - 다중분류가 아닌 이진분류 방식으로 진행
   
- NegativeSampler
  - vocab_size, window_size, sample_size, p(단어별확률) 입력받는 클래스
  - get_negative_sampe
    - 타겟 단어가 주어지면, negative sample 반환
  
## skip_gram.py
- [타겟단어, 문맥단어1, 문맥단어2...] 형태로 입력값 받음
- "타겟단어"와 "문맥단어" 각각 임베딩 층 구성.
  - 임베딩 레이어 통과 후, 나온 은닉 상태 값을 내적한 후 평균 취해주는 skip-gram 방식으로 학습 진행

## trainer.py
- 전체 데이터를 입력받은 후, _batchify 함수 통해 배치 형태로 변환

## train.py
- argument
  - model_fn : 모델 저장 위치
  - train_fn : 학습에 쓰일 데이터 위치
  - hidden_size : 단어 벡터 차원
  - sample_size : negative sampling에 쓰일 샘플 개수
  - window_size
  - gpu_id : gpu 사용 여부
  - n_epochs
  - batch_size
  - verbose
