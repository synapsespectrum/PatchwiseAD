echo "Training Start"
python run.py \
  --is_training=1 \
  --is_pretrain=0 \
  --is_finetune=1 \
  --model=TFADFormer \
  --dataset=SMAP \
  --soft_replacing 0.5 \
  --flip_replacing_interval all \
  --uniform_replacing 0.15 \
  --peak_noising 0.15 \
  --patch_size 4 \
  --batch_size 16 \
  --train_epochs 20

