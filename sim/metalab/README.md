# sim/metalab — 셋업 · 재현성

설치·첫 실행은 리포 루트 [`README.md`](../../README.md). 여기는 setup 이 하는 일과 재현성 모델만.

## `setup.sh` 가 하는 일

1. **핀 고정된 엔진 소스를 형제 디렉터리로 clone** — 커밋은 [`sim_versions.env`](sim_versions.env) 가
   단일 소스다.

   ```
   metalab_ws/
   ├── metalab/         ← 이 리포
   ├── newton/          ← 핀 커밋
   └── genesis-world/   ← 핀 커밋
   ```

2. **엔진별 uv 환경 `uv sync`** — 커밋된 [`setup/<engine>/`](setup/genesis)`{pyproject.toml,uv.lock}` 을
   `~/.metalab/venvs/<repo-dir>/<engine>` 로. lock 이 모든 의존성을 버전+해시로 고정한다(conda 없음).
   엔진 소스는 editable 설치라 바로 고칠 수 있다.

몇 번 돌려도 안전하고(idempotent), `uv` 가 없으면 알아서 설치한다. 첫 `metalab_train.sh` 도 같은 일을
자동으로 하므로 새 머신에서 곧바로 학습 명령부터 시작해도 된다.

## 재현성 모델

런을 결정하는 네 가지가 전부 버전 관리에 있다 — "clone + `setup.sh`" 가 어느 머신에서든 같은 셋업을 만든다.

| 조각 | 위치 | 고정 방식 |
|---|---|---|
| 학습·env·태스크 코드 | 이 리포 | git SHA (런 이름에 포함) |
| 시뮬레이터 소스 (newton, genesis) | 형제 리포 | `sim_versions.env` |
| python 의존성 | 엔진별 `uv.lock` (리포 안) | 버전 + 해시 |
| 로봇·물체 에셋 | `sim/metalab/assets/` (plain git) | git SHA |

> 셋업 재현 = 보장. 비트 단위 동일 결과 = GPU 모델·드라이버·CUDA 가 다르면 보장 안 됨(GPU 비결정성).
> newton 은 NVIDIA 인덱스의 warp-lang dev 빌드를 해시로 핀한다 — 그 nightly 가 인덱스에 남아 있는 한 재현된다.

## 엔진 버전 올리기 (env 담당자)

1. `sim_versions.env` 의 `*_REF` 를 새 커밋으로
2. `setup.sh` 재실행(형제 리포 re-checkout) + `setup/<engine>` 에서 `uv lock`
3. `sim_versions.env` + `uv.lock` 커밋 → 다른 사람은 `setup.sh` 재실행으로 수렴

## Newton RTX 뷰어 (opt-in)

`--viz gl` = OpenGL 창, `--viz rtx` = OVRTX(경로추적, RTX GPU 필요). `ovrtx` 는 기본 env 에 없고
`metalab_train.sh --viz rtx` 가 자동으로 `rtx` extra 를 sync 한다. 손으로 넣으려면:

```bash
UV_PROJECT_ENVIRONMENT=~/.metalab/venvs/<repo-dir>/newton \
  uv sync --project sim/metalab/setup/newton --extra rtx
```
