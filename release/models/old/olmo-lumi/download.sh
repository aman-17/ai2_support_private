
RUN_NAME=$1
STEP_NUM=$2
OUTPUT=$3

aws s3 cp --profile=r2 --endpoint-url=https://a198dc34621661a1a66a02d6eb7c4dc3.r2.cloudflarestorage.com s3://olmo-checkpoints/ai2-llm/olmo-medium/$RUN_NAME/step${STEP_NUM}-unsharded/config.yaml $OUTPUT/step${STEP_NUM}/config.yaml

aws s3 cp --profile=r2 --endpoint-url=https://a198dc34621661a1a66a02d6eb7c4dc3.r2.cloudflarestorage.com s3://olmo-checkpoints/ai2-llm/olmo-medium/$RUN_NAME/step${STEP_NUM}-unsharded/model.pt $OUTPUT/step${STEP_NUM}/model.pt

#aws s3 cp --profile=r2 --endpoint-url=https://a198dc34621661a1a66a02d6eb7c4dc3.r2.cloudflarestorage.com s3://olmo-checkpoints/ai2-llm/olmo-medium/$RUN_NAME/step${STEP_NUM}-unsharded/train.pt $OUTPUT/step${STEP_NUM}/train.pt

echo 'done'
