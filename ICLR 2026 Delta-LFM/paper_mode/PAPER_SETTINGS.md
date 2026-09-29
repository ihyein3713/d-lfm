# 논문 버전 설정 정리 (Delta-LFM, ICLR 2026, arXiv 2512.09185v4)

레포 기준 벤치마크(완료, `../RECIPE.md`)와 논문 기준 벤치마크의 차이를 항목별로 정리한 문서입니다.
근거는 모두 `2512.09185v4.pdf`이며, 쪽/절을 함께 적었습니다. 논문에 없는 값은 **명시 없음**으로
표시하고 임의로 채우지 않았습니다.

- P: 논문에 명시된 값
- R: 레포 기준 실행에서 우리가 쓴 값 (완료된 벤치마크)
- 판정: `그대로` / `플래그만 변경` / `코드 수정 필요` / `결정 필요`

---

## 1. 데이터

| 항목 | P (논문) | R (레포 실행) | 판정 |
|---|---|---|---|
| 코호트 | ADNI, OASIS-3, AIBL 3종 (4.1) | ADNI만 | 그대로 (팀 벤치마크가 ADNI) |
| 전처리 | N4 bias correction, skull stripping, 1.5 mm 등방 resample, 조직 대비 정규화, 표준 템플릿 정합 (4.1) | 동일 파이프라인 결과 사용 | 그대로 |
| split | random **80/5/15** train/val/test (4.1) | 팀 공용 split: subject 116/18/24 = 73/11/15% | **결정 필요 (D1)** |
| 입력 해상도 | crop **64^3** (부록 C) | 전체 뇌 128x144x128을 112^3으로 resize (`--res_scale 0.82`) | **코드 수정 필요** |
| 예측 구간 | 1~13년까지 평가 (부록 D, Table 6) | 15년 미만으로 제한 (`num_class_embeds=15`, `--max_dt_months 180`) | **코드 수정 필요 (아래 4절)** |

## 2. 오토인코더 (AE)

| 항목 | P | R | 판정 |
|---|---|---|---|
| 사전학습 | **사용하지 않음** (부록 C: 베이스라인과의 공정 비교를 위해 미사용, 단 MAISI 초기화 시 PSNR 30.04 -> 31.97) | MAISI `autoencoder_epoch273.pt`로 초기화 | **결정 필요 (D2)** |
| 인코더 채널 | **[64, 128, 256]** (부록 C, 기본값) | MAISI 구조 고정 | **코드 수정 필요** |
| crop 크기 | **64^3** (부록 C, 기본값. 72^3은 A6000 48GB에서 OOM) | 112^3 whole-brain resize | **코드 수정 필요** |
| optimizer | AdamW (부록 A) | AdamW | 그대로 |
| learning rate | **1e-3** (부록 A) | 1e-5 | 플래그만 변경 (`--lr`) |
| batch size | **2** (부록 A) | 1 (+ activation checkpointing) | 플래그만 변경 (`--batch_size`) |
| epochs | **300** (부록 A) | 30 epochs x 2000 steps = 60k steps | **결정 필요 (D3)** |
| ArcRank 손실 | **켬** (3.2, 핵심 기여) | **끔** (`ARCRANK=0`) | **코드 수정 필요 (아래 3절)** |
| 판별자(adversarial) | 언급 없음 | `--use_adv`, warmup 400 / ramp 1200 | **결정 필요 (D4)** |
| 변화재구성 항 | 언급 없음 | `--ae_chgrec_w 0.3` (활성), `--ae_dircosE_w 0.3` (외부 파일 부재로 비활성) | **결정 필요 (D4)** |

## 3. ArcRank 손실 (3.2, 논문 핵심)

논문 정의:

```
SVD(z) = U S V^T   ->  각도 = U,  크기 = S
L_Arc  = sum_{i<j, 같은 환자} |U_i - U_j|
L_Rank = sum_{i<j, 같은 환자} max(0, m - (S_j - S_i)),  t_i < t_j
L_Pull = |S_j - S_i|                      (인접 시점의 과도한 분리 억제)
L_Rank~ = L_Rank + L_Pull                 (가중치는 L_Rank와 공유)
L_ArcRank = lambda_arc * L_Arc + lambda_rank * L_Rank~
안정화: 각 쌍 (i,j)에서 i 쪽 표현에 stop-gradient sg(.) 적용
```

레포 구현(`src/trajectory_losses.py`)은 같은 아이디어를 다르게 구현한 변형입니다.

| 논문 요소 | 레포 대응 | 일치? |
|---|---|---|
| L_Arc = sum \|U_i - U_j\| | `T1` = 1 - line_r2 (rank-1 비율), `C3` = 인접 step 코사인 | 다름 |
| L_Rank 힌지 + margin m | `C1` = PC1 투영의 순서 힌지 (margin 인자 있음) | 부분 일치 |
| L_Pull | 없음 | **없음** |
| stop-gradient sg(i) | 없음 | **없음** |
| lambda_arc, lambda_rank, m | `--angle_weight 0.35`, `--arc_order_w 0.2`, `margin=0.1` | 값 대응 불명 |

- **lambda_arc, lambda_rank, margin m 의 구체적 수치는 논문에 명시 없음.**
- 판정: **코드 수정 필요**. `paper_mode/src/arcrank_loss.py`로 논문 수식을 그대로 구현하고,
  step1 사본에서 이 손실을 호출하는 방식으로 갑니다. 가중치는 **결정 필요 (D5)**.

## 4. Flow Matching

| 항목 | P | R | 판정 |
|---|---|---|---|
| 목표(velocity) | **v\* = (z_j - z_i) / (t_j - t_i)** (식 12) | `--fm_x0_pred 1`: delta = z1 - z0 자체를 회귀 (Dt로 나누지 않음) | 플래그만 변경 가능: `--fm_scheme realtime --fm_x0_pred 0` (레포에 realtime = `(x1-x0)/Dt` 구현 존재) |
| 시간 구간 | **Temporal Sampling [0, T]** (3.3) | `[0,1]` 정규화 경로 | 위와 동일 플래그 |
| 추론 | Euler 적분, **dt = 0.01**, N = (t_j - t_i)/dt (식 14, 15) | 50 step 고정 (`--infer_steps 50`) | 플래그만 변경 (`--infer_steps`) 또는 코드 수정 |
| 시간 임베딩 | **연속 sinusoidal** 인코딩 (부록 B) | **이산 임베딩**: `class_labels = int(dt_years)`, `num_class_embeds=15` (15년 이상 불가) | **코드 수정 필요** |
| 조건 신호 | start time, current sample time, **end query time**, 환자 속성(성별, 나이, 임상상태) (부록 B) | `context` 8채널(나이/성별/진단/부피 5개) + `cond_vec` + 이산 dt | **코드 수정 필요** |
| 조건 주입 방식 | **AdaLN** (부록 B: additive / cross-attention / AdaLN 중 AdaLN이 최적) | additive bias: `emb = emb + cond_mlp(cond_vec)` (`src/model3D/unet.py:1806`) | **코드 수정 필요** |
| 임상상태 잡음 | 학습 시 평균 0, **표준편차 = 클래스 간격/3 (간격=3)** 가우시안 잡음 주입 (부록 F) | 없음 | **코드 수정 필요** |
| 과거 방문 조건 | 언급 없음 (baseline 1장 + 속성) | `--fm_hist_mode prev1+prev2` (이전 2회 방문을 추가 채널로) | **결정 필요 (D6)** |
| 2단계 학습 | 언급 없음 (단일 학습) | base 30 ep + fine-tune 10 ep (energy mask, soft cF1) | **결정 필요 (D6)** |
| learning rate | **3e-5** (부록 A) | 2.5e-5 | 플래그만 변경 |
| batch size | **4** (부록 A) | 8 | 플래그만 변경 |
| epochs | **200** (부록 A) | 30 + 10 | 플래그만 변경 (D3과 함께) |
| 초기 노이즈 sigma | 언급 없음 (식 4의 sigma는 SDE 확장 설명용) | probe로 측정: 0.334 | 그대로 (측정값 유지) |

## 5. 평가 지표

| 지표 | P 정의 | R 구현 | 판정 |
|---|---|---|---|
| PSNR, SSIM | 이미지 레벨 (Table 1). SSIM은 x1e2로 보고 | 있음 (latent-decoded / raw 두 공간) | 그대로 |
| Region MAE | **해마, 편도체, 측뇌실, CSF, 시상** 5개 구조의 MAE (4.1, BrLP 방식) | **없음** (같은 이름의 컬럼은 조건 입력용) | **코드 수정 필요** |
| Delta-RMAE | `\|Dgt - Dgen\|_1 / (0.5 * (\|Dgt\| + \|Dgen\|))`, 범위 [0,2] (식 17) | `DRMAE = \|Dgt - Dgen\|_1 / \|Dgt\|_1` -> **분모가 다름** | **코드 수정 필요** |
| cF1, rF1, dF1 | 없음 (레포 전용 지표) | 있음 | 논문 표에는 미포함, 참고용으로 계속 산출 |

Region MAE에 필요한 분할 라벨은 이미 확보돼 있습니다. `Image/<subject>/<date>/seg.nii.gz`가
FreeSurfer 라벨 체계이고, 753개 전 방문에 존재합니다.

| 구조 | 라벨 |
|---|---|
| 해마 | 17, 53 |
| 편도체 | 18, 54 |
| 측뇌실 | 4, 43 |
| CSF | 24 |
| 시상 | 10, 49 |

## 6. 논문 수치 (비교 목표)

ADNI 기준 Delta-LFM: PSNR **30.59 +- 0.89**, SSIM **94.62 +- 0.85**, Region MAE **0.210 +- 0.28**,
Delta-RMAE **0.436 +- 0.08** (Table 1, 2). 부록 D에 1~13년 구간별 수치가 있어 예측 구간별 비교도
가능합니다.

Ablation(Table 3, 3개 데이터셋 평균)에서 ArcRank + [0,T]가 PSNR 30.04 / SSIM 92.63으로 최고이며,
ArcRank 없는 [0,T]는 28.78 / 90.97입니다. 즉 **논문이 주장하는 이득의 상당 부분이 ArcRank에서
나오고, 우리는 레포 기준 실행에서 그것을 끈 상태로 측정했습니다.**

---

## 결정이 필요한 항목

| ID | 항목 | 선택지 | 권장 |
|---|---|---|---|
| **D1** | split | (a) 팀 공용 split 유지 (b) 논문대로 80/5/15 재추출 | **(a)**. 팀원들과 같은 test set을 써야 비교가 성립하고, 레포 버전 결과와도 직접 비교됩니다 |
| **D2** | MAISI 사전학습 초기화 | (a) 논문 기본대로 미사용 (b) 사용 | **(b) 사용**. 논문도 부록 C에서 MAISI 초기화가 더 좋다고 밝히며(30.04 -> 31.97) 미사용은 베이스라인 공정성 때문입니다. 우리는 베이스라인을 재현하는 게 아니므로 (b)가 성능 비교에 적합하고, 레포 버전과 조건도 일치합니다. 다만 "논문 기본 설정 재현"을 엄격히 원하시면 (a) |
| **D3** | epoch 수 (AE 300 / FM 200) | (a) 논문 숫자 그대로 (b) 총 step 수를 레포 실행과 맞춤 | **(a)**. 단 논문은 1 epoch의 step 수를 명시하지 않습니다. 데이터 1회 순회로 해석하면 AE 300 ep = 약 81k step (레포 60k와 동급, 약 17시간), FM 200 ep = 약 66k step (레포 5k의 13배, 약 29시간)입니다. FM 쪽이 GPU 1장으로 하루를 넘깁니다 |
| **D4** | 레포 추가 손실 (adversarial, ae_chgrec_w) | (a) 끔 (논문 미언급) (b) 켬 | **(a) 끔**. 논문에 없는 항이 들어가면 "논문 설정"이 아니게 됩니다 |
| **D5** | lambda_arc, lambda_rank, margin m | 논문 명시 없음 | 레포 값(0.35 / 0.2 / margin 0.1)을 출발점으로 쓰고 val PSNR로 확인. **임의 결정 사항이라 승인 필요** |
| **D6** | 레포 전용 FM 요소 (prev1+prev2 과거조건, 2단계 fine-tune) | (a) 끔 (b) 켬 | **(a) 끔**. D4와 같은 이유 |
| **D7** | crop 64^3 채택 여부 | (a) 논문대로 64^3 patch + AE 채널 [64,128,256] (b) 레포의 112^3 whole-brain 유지 | **(a)**. 논문 설정의 핵심 중 하나이고 부록 C가 근거를 제시합니다. 단 AE 구조 자체를 새로 만들어야 해서 작업량이 가장 큽니다 |
| **D8** | 평가 공간 | (a) latent-decoded (b) 원본 스캔 공간 | **(a)**. 논문은 디코딩된 이미지에서 PSNR/SSIM을 재며, 우리 레포 버전 결과도 두 공간 모두 있어 비교 가능합니다 |

## 작업량 요약

| 작업 | 종류 | 비고 |
|---|---|---|
| Delta-RMAE, Region MAE 구현 | 신규 파일 | `paper_mode/src/paper_metrics.py`. GPU 불필요, 기존 예측 dump에 바로 적용 가능 |
| ArcRank 논문 수식 구현 | 신규 파일 + step1 사본 수정 | `paper_mode/src/arcrank_loss.py` |
| 연속 시간 임베딩 + AdaLN 조건 | unet 사본 수정 | 가장 침습적. `num_class_embeds` 경로를 대체 |
| AE 구조 [64,128,256] + 64^3 crop | 신규 AE 정의 | MAISI 구조 대체 |
| 임상상태 잡음 주입 | step3 사본 수정 | 표준편차 = 1.0 (간격 3의 1/3) |
| 학습 스크립트 | 신규 | `paper_mode/scripts/` |

## 격리 규칙

- 논문 버전 산출물은 전부 `paper_mode/` 아래에만 둡니다. 레포 코드(`../*.py`, `../src`, `../scripts`,
  `../config`)와 레포 버전 결과(`../../outputs`, `../ae_runs`)는 **읽기만** 합니다.
- 레포 파일을 수정해야 하면 `paper_mode/src/` 또는 `paper_mode/scripts/`에 사본을 만든 뒤 사본만
  고칩니다. 사본 파일 머리에 원본 경로와 변경 요지를 주석으로 적습니다.
- 학습/평가 출력은 `paper_mode/outputs/`를 루트로 씁니다 (레포 버전의 `outputs/`와 분리).

---

# 실행 기록 (2026-09-27 승인분)

D1~D8을 권장안대로 확정하고, D5는 위임받아 두 가지 설정으로 정했습니다.

| ID | 확정 | 비고 |
|---|---|---|
| D1 | 팀 공용 split | `--split_col split --eval_split val` |
| D2 | MAISI 초기화 사용 | 아래 [발견 1] 때문에 D7과 충돌하지 않습니다 |
| D3 | 논문 숫자 (AE 300 epochs) | 1 epoch = train triplet 306개 1회 순회 = 153 step (batch 2) |
| D4 | 레포 추가 손실 끔 | `--no-use_adv --ae_chgrec_w 0 --ae_dircosE_w 0`, 레포 궤적항 전부 0 |
| D5 | 두 설정 병행 (아래) | GPU 4 = A, GPU 5 = B |
| D6 | 레포 전용 FM 요소 끔 | FM 단계에서 적용 |
| D7 | crop 64^3 | `image_size [64,64,64]` + `--crop_mode patch`, `--res_scale` 미사용 |
| D8 | latent-decoded 공간 | |

## 발견 1: 논문의 AE 용량이 곧 MAISI 구조입니다

레포가 인스턴스화하는 MAISI 오토인코더의 `num_channels`가 이미 **[64, 128, 256]**입니다
(`src/autoencoder/maisi/inference_args.pkl`). 논문 부록 C의 기본 용량과 동일합니다. 따라서 D2(MAISI
가중치 초기화)와 D7(채널 [64,128,256])은 서로 모순이 아니고, **구조를 새로 만들 필요도 없습니다.**
레포 실행과의 유일한 차이는 입력 크기입니다: 112^3 whole-brain resize -> 64^3 native patch.
논문이 "MAISI로 초기화하면 [64,128,256]에서 PSNR 30.04 -> 31.97"이라고 쓴 것도 같은 구조를
전제한 서술로 읽힙니다.

## D5: ArcRank 가중치 두 설정

논문에 lambda_arc, lambda_rank, margin m 수치가 없어 다음과 같이 정했습니다.

| 설정 | lambda_arc | lambda_rank | margin | 근거 |
|---|---|---|---|---|
| **A** (`ae_paperA`, GPU 4) | 0.35 | 0.2 | 0.05 | 레포가 같은 역할의 항에 쓰던 값(`--angle_weight 0.35`, `--arc_order_w 0.2`)을 그대로 옮긴 것. 레포 버전과 가중치 규모를 맞춰 비교가 쉽습니다 |
| **B** (`ae_paperB`, GPU 5) | 1.0 | 1.0 | 0.02 | 두 항을 동등 가중하고 margin을 절반 이하로 낮춘 설정. 논문은 두 항에 별도 우선순위를 두지 않으므로 동등 가중이 중립적 출발점이고, margin을 낮추면 hinge가 쉽게 만족돼 pull 항이 상대적으로 강하게 작용합니다 |

margin은 **환자별 평균 latent 크기에 대한 비율**로 구현했습니다(논문은 raw latent 단위의 m을 쓰지만
그 값은 오토인코더 스케일에 의존해 이식이 불가능합니다). A의 0.05는 "다음 방문의 latent 크기가 평균의
5%만큼 커져야 한다"는 뜻입니다. 이 구현 선택은 `src/arcrank_paper.py` docstring에 정리했습니다.

## 실행 중 확정한 값

| 항목 | 값 | 이유 |
|---|---|---|
| 정밀도 | bf16 | 레포 버전 AE도 bf16이었으므로 두 실행을 비교 가능하게 유지. 논문은 정밀도를 명시하지 않음. `AE_PRECISION=fp32`로 변경 가능 |
| 증강 | `--aug_rigid 3 --aug_rot_deg 5` 유지 | 논문 미언급. triplet 공유 rigid 증강은 변화량을 보존하는 데이터 파이프라인 기본값이라 유지 |
| 체크포인트 | `--save_every 10` | 300 epochs면 매 epoch 저장 시 300개가 되어 10 epoch 간격으로 조정 |
| 궤적 프로브 | `--eval_every 25` | ArcRank가 궤적 지표를 실제로 바꾸는지 12회 관측 |

## 실측 소요 시간 (D3 관련, 앞선 추정 수정)

실측 3.0 s/step, 153 step/epoch -> **epoch 당 약 7.7분, 300 epochs = 약 38시간** (GPU 1장, 두 작업
동시 실행 상태). 앞서 17시간으로 추정했던 것보다 깁니다. 64^3인데도 느린 이유는 batch 2가 triplet
3장을 묶어 step 당 볼륨 6개를 처리하고 perceptual loss가 함께 계산되기 때문이며, GPU 사용률은 97%로
데이터 로딩 병목은 아닙니다. 10 epoch 간격 체크포인트가 남으므로 중간에 멈추고 이어받는 판단이
가능합니다.

## 새로 생긴 미결 항목

| ID | 내용 | 상태 |
|---|---|---|
| **M1** | Region MAE의 정의. 논문은 "region-level MAE"라고만 쓰고 복셀 강도 MAE인지 구조 부피 오차인지 밝히지 않습니다. 부피 버전은 생성 영상을 다시 분할해야 하므로(SynthSeg 재실행) 지금은 **기준 방문 seg 마스크 안의 복셀 강도 MAE**로 구현했습니다 | 결정 필요 |
| **M2** | Delta-RMAE를 복셀별 평균으로 볼지 L1 합의 비로 볼지. 부록 E는 "voxel-wise"라고 쓰지만, 실제 MRI 잔차는 거의 모든 복셀에서 0이 아니어서 복셀별 평균은 2에 붙고 논문의 0.436이 나올 수 없습니다. 따라서 **L1 합의 비를 기본값**으로 하고 복셀별 값도 함께 기록합니다 | 결정 필요 (기본값 제안) |

---

# 다음 단계 준비 (2026-09-28)

AE 학습이 끝나는 즉시 이어서 돌 수 있도록 준비를 마쳤습니다.

## 준비된 것

| 파일 | 역할 |
|---|---|
| `src/step3_paper.py` | step3 사본 (추가 57행 / 치환 7행). `--pm_no_class 1`로 이산 dt 임베딩 제거, `--pm_dx_noise`로 임상상태 잡음 |
| `scripts/run_after_ae_paper.sh` | AE 프로세스 종료를 감시 -> latent 추출 -> FM 학습 200 epochs |
| `scripts/eval_fm_paper.sh` | eval_fm 패널 + 논문 지표 재채점 (3 epoch) |
| `data/derived/*.csv` | dt 상한 없는 페어 테이블 (1945쌍, 기존 1939쌍 + 15년 초과 6쌍) |

## 발견 2: 논문의 flow는 노이즈에서 출발하지 않습니다

레포 레시피는 `--fm_res_noise 1` (Delta-Res-Flow: 노이즈 -> delta)인데, 논문 식 12/15는 **baseline
latent z_i에서 출발해 velocity를 적분**합니다. `src/flow.py`의 `compute_ut`에 이미
`"Delta-LFM real-time (Eq.12): u* = (x1-x0)/dt"`라고 적혀 있고, 이 경로는 `--fm_res_noise 0`일 때만
실행됩니다. 따라서 논문 설정은 `--fm_scheme realtime --fm_res_noise 0 --fm_x0_pred 0`이며,
**sigma probe(레시피 5절)는 논문 모드에 불필요합니다** (스케일할 노이즈가 없음). 체인에서 제거했습니다.

## 발견 3: 15년 상한이 사라집니다

`--pm_no_class 1`이면 `num_class_embeds=None`이 되어 dt가 연속 `cond_vec`로만 들어갑니다. 레포의
`num_class_embeds=15`가 강제하던 14년 상한이 없어지므로, 페어 CSV의 dt 상한도 풀었습니다.
논문 부록 D가 13년 구간까지 평가하는 것과 일치합니다.

## 논문 모드 latent 해상도

`--res_scale` 없이 canonical 128x144x128을 그대로 인코딩합니다 (latent 4x32x36x32). 논문 AE는
64^3 crop으로 학습하지만 학습 crop일 뿐이고, 평가는 전체 영상 기준이며 AE는 완전 합성곱이라
크기 제약이 없습니다. 실제로 paper AE 체크포인트로 128x144x128 인코딩/디코딩이 되는 것을
확인했습니다. 레포 모드의 112^3(28^3 latent)과 다릅니다.

## 추가 미결 항목

| ID | 내용 | 상태 |
|---|---|---|
| **M3** | 임상상태 잡음의 표준편차. 부록 F는 "클래스 간격의 1/3 (간격=3)"이라고만 씁니다. 우리 진단 코딩은 0/0.5/1(간격 0.5)이라 같은 비율이면 sd=0.167입니다. 간격 자체를 3으로 읽으면 sd=1.0이 되어 클래스가 자주 뒤바뀝니다. **기본값 0.1667**로 두고 `DX_NOISE=`로 변경 가능하게 했습니다 | 결정 필요 (기본값 제안) |
| **M4** | 추론 적분 스텝. 논문은 dt=0.01, N=(t_j-t_i)/dt이므로 간격마다 스텝 수가 다릅니다. `eval_fm.py`의 `--infer_steps`는 고정값이라, 가장 긴 간격 기준 **200**을 기본값으로 뒀습니다(짧은 페어는 더 잘게 적분). 페어별 가변 스텝이 필요하면 eval 사본을 만들어야 합니다 | 결정 필요 (기본값 제안) |
| **AdaLN** | 부록 B는 additive/cross-attention/AdaLN 중 AdaLN이 최적이라고 하지만, 레포는 additive(`emb + cond_mlp(cond_vec)`)입니다. 1차 실행은 additive로 가고, AdaLN은 UNet 사본을 만들어 별도 변형으로 비교하는 것을 제안합니다 | 결정 필요 |

---

# 미결 항목 확정 (2026-09-28, "논문에 가깝게")

| ID | 확정 | 구현 |
|---|---|---|
| **M2** Delta-RMAE | 부록 E 표현대로 **복셀별 평균을 논문 기준값으로** 보고하고, L1 합의 비를 함께 기록합니다 | `src/paper_metrics.py`가 두 값을 모두 산출. JSON 키: `dRMAE_paper_voxel`(복셀별), `dRMAE_paper`(합의 비) |
| **M4** 추론 적분 | 식 14 그대로 **케이스별** N = (t_j - t_i)/0.01 | `src/eval_fm_paper.py` (eval_fm 사본, 추가 34행). batch_size 1이므로 페어별로 정확히 계산됩니다 |
| **AdaLN** | 부록 B가 최적이라 한 **AdaLN을 기본값으로** 사용. additive는 `COND_MODE=add`로 비교 가능 | `src/adaln_unet.py`. 조건 벡터가 time embedding에 더해지지 않고, 디코더 3개 레벨(채널 768/512/256) 출력에 `h*(1+gamma)+beta`로 작용. head는 zero-init이라 학습 시작 시 항등 |
| **M1** Region MAE | **불가 (도구 없음)**. 아래 참조 | 현재는 기준 방문 seg 마스크 내 복셀 강도 MAE |

## M1이 막힌 이유

논문은 BrLP를 따른다고 하고, BrLP 방식은 **생성 영상을 다시 분할해 구조 부피를 비교**합니다. 그러려면
`mri_synthseg`(FreeSurfer)가 필요한데, 이 서버에는 설치돼 있지 않습니다. 전처리는 다른 장비에서
수행됐습니다 (`preprocess_log.jsonl`의 입력 경로가 `/mnt/d/...`).

- 지금 값: 기준 방문 seg 마스크 안의 **복셀 강도 MAE** (5개 구조 + 평균). 논문 수치와 직접 비교 불가.
- 논문 방식으로 가려면 FreeSurfer 설치(약 10GB, GPU 필요) 후 예측 영상 372장을 분할해야 합니다.
  1장당 GPU 약 1분이므로 체크포인트 하나에 6시간 남짓입니다. **설치 여부는 결정이 필요합니다.**

## AdaLN 검증 시 주의점 (기록)

레포 UNet의 출력 컨볼루션이 zero-init(`zero_module`, `src/model3D/unet.py:1744`)입니다. 따라서
학습 전 모델은 입력과 무관하게 **정확히 0을 출력**하므로, 최종 출력 비교로는 AdaLN 동작을 확인할 수
없습니다(처음에 이 때문에 "효과 없음"으로 잘못 판단했습니다). 검증은 디코더 특징값에서 해야 하며,
`python paper_mode/src/adaln_unet.py`가 그 자기 테스트입니다.
