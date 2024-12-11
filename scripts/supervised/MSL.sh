echo "Training Start"
python run.py \
  --is_training=1 \
  --is_pretrain=0 \
  --is_finetune=1 \
  --model=TFADFormer \
  --dataset=MSL \
  --soft_replacing 0.5 \
  --flip_replacing_interval horizontal \
  --uniform_replacing 0.15 \
  --peak_noising 0.15 \
  --patch_size 2 \
  --batch_size 16 \
  --train_epochs 42
