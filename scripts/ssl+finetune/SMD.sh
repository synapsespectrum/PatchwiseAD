echo "Pretraining Start"

python run.py \
  --is_training=1 \
  --is_pretrain=1 \
  --is_finetune=0 \
  --model=TFADFormer \
  --dataset=SMD \
  --augment 0 \
  --logs=logs/pretrainSMD \
  --batch_size 64 \
  --patch_size 4 \
  --train_epochs 10

echo "Pretraining Done"
echo "===================================================="

echo "Fine-tuning Start"

python run.py \
  --is_training=1 \
  --is_pretrain=0 \
  --is_finetune=1 \
  --model=TFADFormer \
  --dataset=SMD \
  --pretrain_model_path=logs/pretrain_SMD_old/state_dict.pt  \
  --soft_replacing 0.5 \
  --flip_replacing_interval all \
  --uniform_replacing 0.15 \
  --peak_noising 0.15 \
  --length_adjusting 0.1 \
  --batch_size 16 \
  --patch_size 4 \
  --train_epochs 15