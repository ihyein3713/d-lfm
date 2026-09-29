# 실행 기록

## 2026-09-27 00:20 ~ 진행 중: 논문 설정 AE 두 개 + 레포 결과 재채점

### 1. 논문 설정 오토인코더 (D5 두 설정 병행)

| | GPU | 출력 | ArcRank | 상태 |
|---|---|---|---|---|
| A | 4 | `outputs/ae_paperA/` | arc 0.35 / rank 0.2 / margin 0.05 | 학습 중 |
| B | 5 | `outputs/ae_paperB/` | arc 1.0 / rank 1.0 / margin 0.02 | 학습 중 |

명령: `TAG=... PM_ARC=... PM_RANK=... PM_MARGIN=... GPU=... bash scripts/train_ae_paper.sh`
로그: `outputs/ae_paperA.log`, `outputs/ae_paperB.log`

설정은 `../PAPER_SETTINGS.md`의 실행 기록 절을 따릅니다. 논문값: AdamW, lr 1e-3, batch 2,
300 epochs, 64^3 crop, MAISI 구조([64,128,256]). 레포 추가 손실은 모두 꺼져 있습니다
(`[switches] active traj terms = none` 로그로 확인).

첫 스텝에서 관찰한 점: lr 1e-3은 레포 실행의 100배여서 MAISI 사전학습 가중치가 초기 몇 스텝 안에
사실상 리셋됩니다 (재구성 손실 0.037 -> 0.2 대, PSNR 13.7까지 하락 후 상승). 논문은 같은 lr로
300 epochs를 돌리므로 예상된 거동이지만, 실제 회복 여부는 궤적 프로브(25 epoch 간격)로 확인해야
합니다.

### 2. 레포 버전 결과의 논문 지표 재채점

`scripts/rescore_repo_run.sh` (GPU 4, 로그 `outputs/rescore.log`).
완료된 6개 평가(fine-tune ep7/8/9, base ep27/28/29)를 모델 재실행 없이 다시 채점합니다.
eval_fm이 저장해 둔 latent 변화량(delta_*.npy)과 방문별 latent로 `decode(z0 + delta)`를 복원하는
방식이라 체크포인트당 약 17분입니다. 결과는 `outputs/paper_metrics/<tag>.json`.

각 결과에 self-check가 들어갑니다. 복원한 영상으로 다시 계산한 PSNR/SSIM을 원래 결과 JSON의 값과
비교하므로, 값이 어긋나면 복원 경로가 틀린 것입니다.

6쌍 예비 실행값(참고용, 전체 아님): Delta-RMAE 1.19 (copy baseline 2.0). 논문 ADNI 값은 0.436.

### 산출물 위치

```
paper_mode/
  src/arcrank_paper.py            논문 3.2절 ArcRank (자체 테스트 포함)
  src/paper_metrics.py            Delta-RMAE, Region MAE (자체 테스트 포함)
  src/paper_metrics_from_dump.py  완료된 평가 재채점
  src/step1_paper.py              step1_v2_axes.py 사본 (추가 45행, 삭제 0행)
  config/Step1_paper.yaml         64^3 crop
  scripts/train_ae_paper.sh       논문 설정 AE 학습
  scripts/rescore_repo_run.sh     재채점 일괄 실행
  outputs/                        전 산출물 (레포 버전 outputs와 분리)
```

레포 파일은 하나도 수정하지 않았습니다. `step1_paper.py`는 사본이며 원본과의 차이는 헤더,
sys.path 설정, `--pm_*` 인자 처리, ArcRank 호출 블록뿐입니다 (추가 45행 / 삭제 0행).

## 2026-09-28 00:00~00:35: 다음 단계 준비 완료

AE 두 개(A/B)는 epoch 165/300 지점에서 정상 진행 중이며, 남은 135 epoch는 약 19시간
(epoch 당 8.6분 실측)으로 9/28 19:30경 종료 예정입니다. 그 사이에 다음 단계를 전부 준비했습니다.

### 준비 항목과 검증 상태

| 항목 | 검증 |
|---|---|
| `data/derived/` 페어 CSV (dt 상한 없음) | 1945쌍 생성 확인 (레포 모드 1939쌍 + 15년 초과 6쌍) |
| `src/step3_paper.py` | 스모크 통과: `[paper_mode] num_class_embeds=None` 출력 후 학습 진입, Loss=0.0146 |
| `scripts/run_after_ae_paper.sh` | 문법 검사 통과. AE 프로세스 종료 감시 -> latent -> FM |
| `scripts/eval_fm_paper.sh` | 문법 검사 통과 |
| paper AE로 canonical 인코딩 | `all-ae-160-3D.pth` 로드 후 128x144x128 -> latent 4x32x36x32 확인 |
| Region MAE seg 정렬 | seg가 121x145x121이라 canonical로 pad/crop 후 계산하도록 수정, 5개 구조 전부 산출 확인 |
| 재채점 self-check | base ep27에서 PSNR 차이 0.0011 dB, SSIM 차이 0.0 -> 복원 경로가 정확함 |

### 재채점 재실행

Region MAE를 포함해 6개 태그를 다시 채점 중입니다 (`outputs/rescore2.log`). 이전 실행분은
Region MAE가 빠져 있었습니다 (seg 형태 불일치로 건너뜀).

### AE 종료 후 실행할 명령

```bash
cd "/mnt/aix22308/longi_v2/d_lfm/ICLR 2026 Delta-LFM/paper_mode"
TAG=ae_paperA GPU=4 nohup bash scripts/run_after_ae_paper.sh > outputs/chain_A.out 2>&1 &
TAG=ae_paperB GPU=5 nohup bash scripts/run_after_ae_paper.sh > outputs/chain_B.out 2>&1 &
```

두 스크립트는 각자 자기 TAG의 AE 프로세스가 끝날 때까지 기다리므로 **지금 미리 띄워 둬도**
됩니다. 다만 두 AE가 같은 시각에 끝나면 GPU 4/5에서 동시에 FM 학습이 시작됩니다.

## 2026-09-28 01:00~02:00: 논문 기준으로 미결 항목 확정

`PAPER_SETTINGS.md`의 M2 / M4 / AdaLN을 논문에 가까운 쪽으로 구현했습니다. M1(Region MAE)은
`mri_synthseg`가 이 서버에 없어 막혀 있습니다.

| 추가된 파일 | 내용 |
|---|---|
| `src/adaln_unet.py` | AdaLN 조건 주입 (부록 B). 자기 테스트 포함 |
| `src/eval_fm_paper.py` | eval_fm 사본 (추가 34행 / 치환 3행). 케이스별 Euler 스텝 N=(t_j-t_i)/0.01 |

체인 기본값이 `COND_MODE=adaln`, `DX_NOISE=0.1667`, `DT_STEP=0.01`로 바뀌었습니다.
additive 비교가 필요하면 `COND_MODE=add`로 같은 체인을 다시 돌리면 됩니다.

## 2026-09-28 09:25: SynthSeg 설치 중 GPU 4의 AE 학습을 죽였습니다

경위: SynthSeg 동작 확인을 GPU 4에서 실행했는데, TensorFlow가 기동 시 카드 메모리를 거의 전부
선점합니다(로그에 `Created TensorFlow device ... with 36580 MB memory`). 같은 카드에서 돌던
`ae_paperA`가 CUDA OOM으로 종료됐습니다 (`[paper_mode] exit: 1`, epoch 221 직후).

조치:
1. SynthSeg 프로세스 종료, GPU 4 반환.
2. `scripts/train_ae_paper.sh`에 `RESUME_CKPT` 추가 (step1의 `--aekl_ckpt`로 가중치만 이어받음.
   optimizer 상태는 트레이너가 저장하지 않으므로 복원되지 않습니다).
3. `ae_paperA/all-ae-220-3D.pth`(epoch 219 종료 시점)에서 **새 TAG `ae_paperA_r220`으로 80 epoch**
   재시작. 새 TAG를 쓴 이유는 epoch 번호가 0부터 다시 매겨져 기존 체크포인트를 덮어쓰기 때문입니다.
   재시작 직후 rec 0.0285로, 중단 시점 수준에서 이어짐을 확인했습니다.
4. 손실분: epoch 220~221 두 번 (체크포인트는 10 epoch 간격이므로 219까지 보존).

재발 방지:
- `src/region_volumes.py`가 SynthSeg를 호출할 때 `TF_FORCE_GPU_ALLOW_GROWTH=1`을 강제합니다.
- 그래도 **다른 작업이 있는 카드에서는 SynthSeg를 돌리지 않습니다.** `--cpu --threads N` 경로를
  마련해 뒀고, GPU가 완전히 빌 때까지는 CPU로 돌립니다.
- AE B(GPU 5)는 영향 없이 계속 진행 중입니다.

## 2026-09-29 01:00~01:35: 논문 FM 체인 시작, 조건 주입 버그 수정

AE 두 개 모두 300 epoch 완료 (A는 GPU 사고로 epoch 219에서 재시작해 총 299).

| | 최종 PSNR / SSIM | line_r2 | mono_gap | step_cos |
|---|---|---|---|---|
| A (arc 0.35 / rank 0.2) | 27.60 / 0.876 | 0.666 | 0.339 | -0.360 |
| B (arc 1.0 / rank 1.0) | 28.41 / 0.899 | 0.674 | 0.354 | -0.384 |
| 레포 AE (ArcRank 없음, 참고) | - | 0.456 | 0.159 | -0.469 |

논문 ArcRank가 직선성과 시간 순서를 크게 개선했습니다. step_cos는 여전히 음수인데, 이 프로브는
64^3 패치 기준이라 112^3에서 측정한 레포 ArcRank 값(+0.33)과 직접 비교할 수 없습니다.

어느 AE를 쓸지 고르지 않고 **두 설정 모두 FM까지 진행**합니다 (GPU 4 = A, GPU 5 = B).

### 버그: 논문 설정에서는 조건이 전혀 들어가지 않았습니다

첫 실행 로그에 `[paper_mode] cond_dim=0: AdaLN has nothing to condition on` 이 찍혔습니다. 원인은
step3의 모델 생성 분기입니다. 연속 조건 벡터(`build_cond_vec`: 시작 나이, 간격, 성별, 진단, 부피
5개)는 `--fm_res_noise 1` 또는 `--fm_x0_cond 1` 분기에서만 `cond_dim`으로 전달되고, 논문 설정이
쓰는 평범한 분기(`in_channels=4`)에서는 `cond_dim`이 0이었습니다. `--pm_no_class 1`로 이산 간격
임베딩까지 꺼둔 상태였으므로, **모델이 dt조차 모르는 상태로 학습되고 있었습니다.**

수정: 사본 두 개(`step3_paper.py`, `eval_fm_paper.py`)의 해당 분기에 `cond_dim=10`을 전달합니다.
스모크로 `[paper_mode] AdaLN conditioning on 3 decoder levels, channels [768, 512, 256]` 출력과
정상 학습(Loss 0.0058)을 확인한 뒤 두 체인을 재시작했습니다. 잘못 학습된 20분치는 버렸습니다.

### Region MAE 예측 export 버그

예측 파일명을 `환자__추적일`로 만들어, 같은 추적 방문을 여러 baseline에서 예측한 경우 서로
덮어썼습니다 (372 -> 99). 키를 `환자__기준일__추적일`로 바꿔 372개가 모두 남습니다. 정답은 추적
방문 단위가 맞으므로 99개 그대로 재사용합니다.
