echo "Training Start"
python run.py \
  --is_training=1 \
  --is_pretrain=0 \
  --is_finetune=1 \
  --model=TFADFormer \
  --dataset=SMD \
  --soft_replacing 0.5 \
  --flip_replacing_interval all \
  --uniform_replacing 0.15 \
  --peak_noising 0.15 \
  --length_adjusting 0.1 \
  --batch_size 16 \
  --patch_size 4 \
  --train_epochs 15