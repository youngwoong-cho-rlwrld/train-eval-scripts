#!/bin/bash
#SBATCH --job-name=eval-onboarding-youngwoong-scratch
#SBATCH --partition=rlwrld
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/rlwrld2/home/youngwoong_cho/logs/%x_%j.out
#SBATCH --error=/rlwrld2/home/youngwoong_cho/logs/%x_%j.err

set -e
export OMNI_KIT_ACCEPT_EULA=Y
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1

# ─── 설정 ───
GROOT_DIR="/rlwrld2/home/youngwoong_cho/workspace/gr00t"
ISAAC_DIR="/rlwrld2/home/youngwoong_cho/workspace/rlwrld_isaac"
CHECKPOINT="/rlwrld2/home/youngwoong_cho/checkpoints/onboard_cube_box_left_scratch/checkpoint-30000"
EVAL_SET="${EVAL_SET:-0cm}"
GPU_LABEL="${GPU_LABEL:-a100}"
TIMESTAMP="$(date +%Y%m%d%H%M%S)"
OUTPUT_DIR="/rlwrld2/home/youngwoong_cho/eval_results/onboard_cube_box_left_scratch/${EVAL_SET}_${GPU_LABEL}_${TIMESTAMP}"
TASK_NAME="task-Cube_Box-5cmLeft"
INSTRUCTION="Pick up the cube with your left hand and place it in the box"
DATA_CONFIG="allex_thetwo_ck40_egostereo"
N_EPISODES=70
EXECUTION_HORIZON=8

mkdir -p "$OUTPUT_DIR"

# ─── 랜덤 포트 선택 (1024~65535 중 사용 중이지 않은 포트) ───
find_available_port() {
    while true; do
        local port=$((RANDOM % 64511 + 1024))
        if ! ss -tuln | grep -q ":$port "; then
            echo $port; return 0
        fi
    done
}
PORT=$(find_available_port)
echo "[port] using $PORT"

# ─── 종료 시 서버 정리 (trap: EXIT/INT/TERM) ───
SERVER_PID=""
cleanup() {
    [ -n "$SERVER_PID" ] && kill -TERM -$SERVER_PID 2>/dev/null || true
    pkill -9 -f "server_v2.py.*--port $PORT" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ─── Isaac Sim 서버 기동 (백그라운드, 새 프로세스 그룹) ───
setsid bash -c "
    source '$ISAAC_DIR/.venv/bin/activate'
    cd '$ISAAC_DIR'
    exec python scripts/environments/server_v2.py \
        --task Isaac-UniPickPlace-ALLEX-JointAction-VisualStereo-Abs-v0 \
        --task-name '$TASK_NAME' \
        --eval-set '$EVAL_SET' \
        --device cpu \
        --port $PORT \
        --image-crop-ratio 1.0 \
        --image-resize-height 480 \
        --image-resize-width 640 \
        --app-launcher.headless
" > "$OUTPUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "[server] pid=$SERVER_PID, log=$OUTPUT_DIR/server.log"

# ─── 서버 준비 대기 (최대 5분, 단 서버가 죽으면 즉시 종료) ───
for i in $(seq 1 150); do
    if ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
        echo "[server] LISTENING on $PORT"; break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[server] DIED before opening port $PORT — last 20 lines of $OUTPUT_DIR/server.log:"
        tail -20 "$OUTPUT_DIR/server.log" 1>&2
        exit 1
    fi
    sleep 2
done
if ! ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
    echo "[server] FAILED to start within 5min"; exit 1
fi
sleep 5   # Isaac 추가 초기화 여유

# ─── 클라이언트 실행 ───
source "$GROOT_DIR/.venv/bin/activate"
cd "$GROOT_DIR"
python scripts/eval_allex.py \
    --model-path "$CHECKPOINT" \
    --server-port $PORT \
    --output-dir "$OUTPUT_DIR" \
    --instruction "$INSTRUCTION" \
    --n-episodes $N_EPISODES \
    --execution-horizon $EXECUTION_HORIZON \
    --data-config "$DATA_CONFIG" \
    --action-type joint_action

echo "[eval] done → $OUTPUT_DIR/results.json"
