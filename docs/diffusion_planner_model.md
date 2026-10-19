# Diffusion Planner 동작 방식 정리

이 문서는 현재 저장소 코드 기준으로 Diffusion Planner가 **어떤 데이터를 입력받아, 어떻게 학습하고, 어떻게 궤적을 생성하는지**를 정리한 것입니다.
기본 하이퍼파라미터 값은 [train_predictor.py](../train_predictor.py)의 argparse 기본값을 기준으로 합니다.

---

## 1. 전체 흐름

```
[nuPlan DB] ──data_process.py──▶ [.npz cache] ──DataLoader──▶ [augmentation + ego-centric 변환]
                                                                       │
                                                                       ▼
                                                         [ObservationNormalizer]
                                                                       │
                       ┌───────────────────────────────────────────────┤
                       ▼                                               ▼
        Encoder (주변 agent 과거 / 정적 객체 / lane)         Decoder (DiT, diffusion)
                       │  scene encoding [B, 107, 192]                 ▲
                       └───────────── cross-attention ─────────────────┘
                                                                       │
                                     학습: x_t → x_0 예측 (MSE)          │
                                     추론: DPM-Solver++ 10 step으로 노이즈 → 궤적
                                                                       ▼
                                   ego(+ 주변 agent 10대) 미래 궤적 [B, 1+10, 80, 4]
```

| 구성 요소 | 파일 |
|---|---|
| 전처리 | [data_processor.py](../diffusion_planner/data_process/data_processor.py), `agent_process.py`, `map_process.py`, `ego_process.py` |
| Dataset | [dataset.py](../diffusion_planner/utils/dataset.py) |
| Augmentation | [data_augmentation.py](../diffusion_planner/utils/data_augmentation.py) |
| Normalizer | [normalizer.py](../diffusion_planner/utils/normalizer.py), [normalization.json](../normalization.json) |
| 모델 | [diffusion_planner.py](../diffusion_planner/model/diffusion_planner.py), [encoder.py](../diffusion_planner/model/module/encoder.py), [decoder.py](../diffusion_planner/model/module/decoder.py), [dit.py](../diffusion_planner/model/module/dit.py) |
| Diffusion | [sde.py](../diffusion_planner/model/diffusion_utils/sde.py), [sampling.py](../diffusion_planner/model/diffusion_utils/sampling.py) |
| Loss / 학습 루프 | [loss.py](../diffusion_planner/loss.py), [train_epoch.py](../diffusion_planner/train_epoch.py), [train_predictor.py](../train_predictor.py) |
| Guidance | [guidance_wrapper.py](../diffusion_planner/model/guidance/guidance_wrapper.py), [collision.py](../diffusion_planner/model/guidance/collision.py) |
| 시뮬레이션 planner | [planner.py](../diffusion_planner/planner/planner.py) |

---

## 2. 시간 축

| 구분 | 길이 | 간격 | 개수 |
|---|---|---|---|
| 과거 (입력) | 2초 | 0.1초 | 20개 + 현재 = **21** (`time_len`) |
| 미래 (예측) | 8초 | 0.1초 | **80** (`future_len`) |

---

## 3. 데이터 (전처리 결과 `.npz`)

모든 좌표는 **현재 시점 ego의 rear axle을 원점, ego heading을 x축**으로 하는 ego-centric 좌표계입니다.

| key | shape | 내용 |
|---|---|---|
| `ego_current_state` | (10,) | x, y, cos, sin, vx, vy, ax, ay, steering angle, yaw rate |
| `ego_agent_future` | (80, 3) | 미래 x, y, heading |
| `neighbor_agents_past` | (32, 21, 11) | x, y, cos, sin, vx, vy, width, length, type one-hot(3: vehicle / pedestrian / bicycle) |
| `neighbor_agents_future` | (32, 80, 3) | 미래 x, y, heading (없는 시점은 0) |
| `static_objects` | (5, 10) | x, y, cos, sin, width, length, type one-hot(4) |
| `lanes` | (70, 20, 12) | 중심선 x, y, 다음 점까지 (dx, dy), 좌/우 경계까지 offset(4), 신호등 one-hot(4) |
| `lanes_speed_limit`, `lanes_has_speed_limit` | (70, 1) | 제한속도, 제한속도 존재 여부 |
| `route_lanes` | (25, 20, 12) | route에 속한 lane (형식은 `lanes`와 동일) |
| `route_lanes_speed_limit`, `route_lanes_has_speed_limit` | (25, 1) | |

선택 규칙:
- **주변 agent**: 현재 시점 거리 순으로 가까운 32대. 단, 보행자와 자전거는 최대 10개까지만 넣습니다.
- **정적 객체**: 가까운 순으로 5개를 넣습니다.
- **lane**: ego 반경 100m 안의 lane입니다. 각 lane은 점 20개로 보간합니다.
- **route lane**: route roadblock에 속한 lane입니다. 끊어진 route는 `route_roadblock_correction`으로 보정합니다.
- 빈 슬롯은 모두 0으로 채웁니다. 모델은 "모든 값이 0인지"로 mask를 판단합니다.

---

## 4. 학습 시 입력 전처리 ([train_epoch.py](../diffusion_planner/train_epoch.py))

### 4.1 State Perturbation (augmentation)

[data_augmentation.py](../diffusion_planner/utils/data_augmentation.py)의 `StatePerturbation`이 처리합니다.

1. 샘플마다 확률 `1 - augment_prob`(기본 0.5)로 적용합니다. 단, ego 속도가 2m/s 이상인 샘플에만 적용합니다.
2. ego 현재 상태에 균등분포 노이즈를 더합니다. 범위는 y ±0.75m, yaw ±0.35rad, vx ±1, vy ±0.5, ax ±0.2, ay ±0.1입니다.
3. 섭동된 현재 상태에서 GT 궤적의 2.1초 지점까지를 **5차 다항식(quintic)**으로 다시 이어서, 미래 궤적의 처음 20개 점(0.1~2.0초)을 교체합니다.
   - 위치, 속도, 가속도의 경계조건을 양쪽 끝에서 맞춥니다.
   - 2.1초 이후는 GT를 그대로 씁니다.
4. 섭동된 ego 위치와 heading을 새 원점으로 삼아 모든 입력과 GT를 다시 ego-centric으로 변환합니다(`centric_transform`).

이렇게 하면 모델은 "약간 벗어난 상태에서 원래 경로로 복귀하는" 궤적을 학습합니다. closed-loop에서 오차가 누적되는 문제(covariate shift)를 완화하려는 목적입니다.

### 4.2 heading → (cos, sin), 정규화

- 미래 궤적 `(x, y, heading)`을 `(x, y, cos, sin)` 4차원으로 바꿉니다.
- 관측 입력은 `ObservationNormalizer`로 정규화합니다. 기준은 [normalization.json](../normalization.json)이고, 예를 들어 x는 `(x - 10) / 20`, y는 `y / 20`입니다. 모든 값이 0인 빈 슬롯은 정규화 후에도 0으로 유지합니다.
- 미래 궤적은 `StateNormalizer`로 같은 방식(mean `[10,0,0,0]`, std `[20,20,1,1]`)으로 정규화합니다.

---

## 5. Encoder ([encoder.py](../diffusion_planner/model/module/encoder.py))

scene 요소를 각각 `hidden_dim = 192` 차원의 token으로 만든 뒤 Transformer로 섞습니다.

| 하위 encoder | 입력 | 처리 | 출력 token 수 |
|---|---|---|---|
| `AgentFusionEncoder` | `neighbor_agents_past` (8 feature + valid flag, 21 step) | MLP-Mixer(시간 축과 채널 축을 번갈아 MLP) × 3 → 시간 평균 pooling → type embedding 더함 | 32 |
| `StaticFusionEncoder` | `static_objects` | MLP | 5 |
| `LaneFusionEncoder` | `lanes` (8 feature, 20 point) | MLP-Mixer × 3 → 점 평균 pooling → 제한속도 / 신호등 embedding 더함 | 70 |

- 각 token에는 **position embedding**을 더합니다. 대표 위치 `(x, y, cos, sin)`과 종류 one-hot(agent / static / lane)을 합친 7차원을 Linear로 투영한 값입니다.
- `FusionEncoder`: 107개 token에 self-attention block(`encoder_depth = 3`)을 적용합니다. 빈 token은 `key_padding_mask`로 가립니다.
- 출력 `encoding`은 `[B, 107, 192]`이고, decoder의 cross-attention context로 쓰입니다.

> **ego 정보는 encoder에 들어가지 않습니다.** ego는 decoder에서 궤적의 첫 점(현재 위치 / heading)으로만 들어갑니다. ego-centric 변환 후에는 현재 위치 / heading이 항상 `(0, 0, cos 0, sin 0)`이므로, 모델은 ego의 속도나 가속도를 직접 보지 않습니다.
> 미래를 단순히 현재 운동 상태의 연장으로 예측하는 지름길(shortcut)을 막기 위한 설계로 보입니다. 코드에서 `ego_current_state`의 속도·가속도 성분(인덱스 4~9)은 augmentation에서만 쓰입니다.

---

## 6. Decoder: Diffusion Transformer ([decoder.py](../diffusion_planner/model/module/decoder.py), [dit.py](../diffusion_planner/model/module/dit.py))

### 6.1 예측 대상 표현

- 예측 대상은 `P = 1 + predicted_neighbor_num`(기본 1 + 10)개의 **agent token**입니다. 0번이 ego이고, 1~10번은 거리 순으로 가까운 주변 agent입니다.
- token 하나는 `(1 + 80) × 4 = 324`차원 벡터, 즉 현재 상태 1개와 미래 80 step의 `(x, y, cos, sin)`을 펼친 것입니다.
- 첫 step(현재 상태)은 diffusion 과정 내내 **실제 값으로 고정**합니다(inpainting 방식).

### 6.2 DiT 구조

```
x [B, P, 324] ──preproj(MLP)──▶ [B, P, 192] + agent embedding(ego / neighbor 2종)
y = RouteEncoder(route_lanes) + TimestepEmbedder(t)          # 조건 벡터 [B, 192]

DiTBlock × decoder_depth(3):
    x = x + gate · SelfAttn( adaLN(x; y) )     # agent token 간 상호작용, 없는 neighbor는 mask
    x = x + gate · MLP( adaLN(x; y) )
    x = CrossAttn( LN(x), encoding )           # scene context 참조
    x = MLP( LN(x) )

FinalLayer: adaLN(x; y) → MLP → [B, P, 324]
```

- **adaLN-Zero**: 조건 `y`(route와 diffusion time)로 LayerNorm의 shift, scale, gate를 만들어 주입합니다. 학습 시작 시점에는 modulation과 출력층을 0으로 초기화합니다.
- **RouteEncoder**: route lane 25 × 20개 점을 MLP-Mixer로 묶어 벡터 하나로 만듭니다. 이 벡터가 "어디로 가야 하는지"를 알려주는 전역 조건입니다.
- **Self-attention**: ego와 주변 agent가 서로의 미래를 참조하는 **joint prediction**의 핵심입니다.
- **Cross-attention**: 107개 scene token(주변 agent 과거, 정적 객체, lane)을 참조합니다.
- `diffusion_model_type`
  - `x_start`(기본): 깨끗한 궤적 `x_0`를 직접 출력합니다.
  - `score`: 출력을 `σ(t)`로 나눠 score를 출력합니다.

---

## 7. Diffusion 과정 ([sde.py](../diffusion_planner/model/diffusion_utils/sde.py))

**VP-SDE (linear β)**: `β(t) = 0.1 + (20 − 0.1)·t`, `t ∈ [0, 1]`

- forward(노이즈 추가): `x_t = α(t)·x_0 + σ(t)·z`, `z ~ N(0, I)`
- `α(t) = exp(−¼ t²(β_max − β_min) − ½ β_min t)`, `σ(t) = √(1 − α(t)²)`

### 7.1 학습 ([loss.py](../diffusion_planner/loss.py))

1. GT를 만듭니다: `x_0 = [현재 상태 | 정규화된 미래 80 step]`, shape `[B, P, 81, 4]`. 존재하지 않는 neighbor나 시점은 0으로 둡니다.
2. 샘플마다 `t ~ U(1e-3, 1)`을 뽑고, 미래 부분에만 노이즈를 더해 `x_t`를 만듭니다. 첫 step은 깨끗한 값을 유지합니다.
3. 모델 출력 `D(x_t, t, scene)`로 loss를 계산합니다.
   - `x_start`: `‖D − x_0‖²`
   - `score`: `‖D·σ + z‖²`
4. 최종 loss:
   ```
   loss = neighbor_prediction_loss + alpha_planning_loss × ego_planning_loss
   ```
   - `ego_planning_loss`: ego token의 모든 미래 step 평균
   - `neighbor_prediction_loss`: neighbor 중 **GT가 있는 step만** 평균

### 7.2 추론 (`Decoder.forward`의 eval 분기)

1. 초기값: `x_T = [현재 상태 | N(0, 0.5²) 노이즈 80 step]`. 표준편차가 1이 아니라 0.5입니다.
2. **DPM-Solver++**(2차, multistep, logSNR 간격) **10 step**으로 `x_T → x_0`를 풉니다([sampling.py](../diffusion_planner/model/diffusion_utils/sampling.py)).
3. 매 step마다 `correcting_xt_fn`으로 첫 step을 현재 상태로 다시 덮어씁니다.
4. 결과를 `StateNormalizer.inverse`로 되돌리고 첫 step을 버립니다. 출력 `prediction`은 `[B, P, 80, 4]`입니다.

### 7.3 Classifier Guidance (선택)

[collision.py](../diffusion_planner/model/guidance/collision.py)의 `collision_guidance_fn`이 담당합니다.

- 예측된 ego bounding box와 **예측된 neighbor** bounding box 사이의 signed distance로 충돌 energy를 계산합니다.
- 이 energy의 gradient로 sampling을 밀어냅니다. scale은 0.5입니다.
- diffusion time `0.005 < t < 0.1`인 마지막 구간에서만 gradient가 흐릅니다.
- config에서 `guidance_fn`이 설정된 경우에만 동작합니다(`diffusion_planner_guidance.yaml`).
- **neighbor 예측을 사용하므로, `predicted_neighbor_num = 0`이면 의미가 없어집니다.**

---

## 8. 학습 설정 ([train_predictor.py](../train_predictor.py))

| 항목 | 기본값 |
|---|---|
| Optimizer | AdamW, lr 5e-4, grad clip 5 |
| LR schedule | 5 epoch 동안 0.1× → 1×로 linear warmup, 이후 **고정**. 함수 이름은 `CosineAnnealingWarmUpRestarts`이지만 실제로는 cosine decay가 없습니다. |
| Batch | 전체 2048 (GPU 수로 나눔), DDP |
| Epoch | 500, 20 epoch마다 저장 |
| EMA | decay 0.999. 추론 시 기본으로 EMA 가중치를 사용합니다. |
| 모델 크기 | hidden 192, head 6, encoder / decoder depth 3, drop path 0.1 |
| 검증 | **없음**. 학습 loss만 기록합니다. |

---

## 9. 시뮬레이션에서의 동작 ([planner.py](../diffusion_planner/planner/planner.py))

nuPlan의 `AbstractPlanner`를 구현한 클래스이며, 시뮬레이션 매 step마다 다음을 수행합니다.

1. `DataProcessor.observation_adapter`가 history buffer, 신호등, 맵, route로 전처리와 같은 형식의 입력을 만듭니다. 학습 데이터와 같은 함수(agent, map 처리)를 사용합니다.
2. `ObservationNormalizer`로 정규화한 뒤 모델 forward를 실행합니다. 이때 encoder는 1회, DiT는 sampling step 수만큼 돕니다.
3. `prediction[0, 0]`, 즉 **ego 궤적만** 꺼냅니다. `atan2(sin, cos)`로 heading을 복원합니다.
4. `transform_predictions_to_states`로 전역 좌표의 `EgoState` 열로 바꾸고, `InterpolatedTrajectory`로 반환합니다. 실제 차량 추종은 nuPlan의 controller가 맡습니다.

주변 agent 예측은 시뮬레이션 출력에 쓰이지 않습니다. 학습 시 joint modeling 효과와, guidance 사용 시 충돌 계산에만 쓰입니다.

---

## 10. 구현상 참고 사항

- **mask 규칙**: 거의 모든 모듈이 "feature가 모두 0인지"로 빈 슬롯을 판단합니다. 실제 값이 우연히 전부 0이면 빈 슬롯으로 처리됩니다. 데이터를 직접 가공할 때 주의해야 합니다.
- **DiTBlock의 cross-attention**: self-attention과 MLP는 residual 구조인데, cross-attention과 그 뒤 MLP(`x = cross_attn(...)`, `x = mlp2(...)`)에는 residual 연결이 없습니다. 구조를 바꿀 때 참고하세요.
- **FusionEncoder**: `mask[:, 0] = False`로 첫 token을 항상 유효하게 둡니다. 모든 token이 mask되면 attention에서 NaN이 나는 것을 막기 위한 처리입니다.
- **`predicted_neighbor_num`**: Dataset, StateNormalizer, Decoder, loss가 모두 이 값을 따릅니다. 0으로 두면 ego만 예측하게 됩니다. 이때 파라미터 shape는 그대로입니다.
