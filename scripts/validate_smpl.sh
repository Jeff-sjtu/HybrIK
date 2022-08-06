CONFIG=$1
CKPT=$2
PORT=${3:-23456}

HOST=$(hostname -i)

python ./scripts/validate_smpl.py \
    --batch 32 \
    --gpus 0,1,2 \
    --world-size 3 \
    --flip-test \
    --launcher pytorch --rank 0 \
    --dist-url tcp://${HOST}:${PORT} \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT}
