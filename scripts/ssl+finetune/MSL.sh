echo "Pretraining Start"
python run.py \
  --is_training=1 \
  --is_pretrain=1 \
  --is_finetune=0 \
  --model=TFADFormer \
  --dataset=MSL \
  --logs=logs/pretrainMSL \
  --patch_size 2 \
  --batch_size 64 \
  --train_epochs 20

echo "Pretraining Done"
echo "===================================================="

echo "Fine-tuning Start"
python run.py \
  --is_training=1 \
  --is_pretrain=0 \
  --is_finetune=1 \
  --model=TFADFormer \
  --dataset=MSL \
  --soft_replacing 0.5 \
  --flip_replacing_interval horizontal \
  --pretrain_model_path=logs/pretrainMSL/state_dict.pt  \
  --uniform_replacing 0.15 \
  --peak_noising 0.15 \
  --patch_size 2 \
  --batch_size 16 \
  --train_epochs 42
