# Yolov5m - From scratch
python val.py \
--weights runs/train/Shoaib/HCK/HCK_Exp4_Yolov5M_v2/Yolov5m_Train_from_scratch_using_full_GT/weights/best.pt \
--device 3 \
--name HCK_Val_Yolov5m_scratchGT_Conf50 \
--data data_HCK/HACKTHON_full.yaml \
--conf 0.5 \
--save-conf \
--save-txt \
--exist-ok



# Yolov5m - Retrain using GT
python val.py \
--weights runs/train/Shoaib/HCK/HCK_Exp4_Yolov5M_v2/Yolov5m_Retrain_using_HCK_Full_GT/weights/best.pt \
--device 3 \
--name HCK_Val_Yolov5m_scratchGT_retrainGT_Conf50 \
--data data_HCK/HACKTHON_full.yaml \
--conf 0.5 \
--save-conf \
--save-txt \
--exist-ok



# Yolov5m - Retrain using goodGT failedPD
python val.py \
--weights runs/train/Shoaib/HCK/HCK_Exp4_Yolov5M_v1/Yolov5m_Retrain_using_HCK_passedGT_failedPD/weights/best.pt \
--device 3 \
--name HCK_Val_Yolov5m_scratchGT_retrainFiltered_Conf50 \
--data data_HCK/HACKTHON_full.yaml \
--conf 0.5 \
--save-conf \
--save-txt \
--exist-ok



# Yolov5m - Retrain using recovered goodGT failedPD
python val.py \
--weights runs/train/Shoaib/HCK/HCK_Exp4_Yolov5M_v2/Yolov5m_Retrain_using_HCK_passedGT_failedPD/weights/best.pt \
--device 3 \
--name HCK_Val_Yolov5m_scratchGT_recovered_retrainFiltered_Conf50 \
--data data_HCK/HACKTHON_full.yaml \
--conf 0.5 \
--save-conf \
--save-txt \
--exist-ok




















# Yolov5m - From scratch
python val.py \
--weights runs/train/Shoaib/HCK/HCK_Exp4_Yolov5M_v2/Yolov5m_Train_from_scratch_using_full_GT/weights/best.pt \
--device 3 \
--name HCK_Val_Yolov5m_scratchGT_Conf01 \
--data data_HCK/HACKTHON_full.yaml \
--save-conf \
--save-txt \
--exist-ok



# Yolov5m - Retrain using GT
python val.py \
--weights runs/train/Shoaib/HCK/HCK_Exp4_Yolov5M_v2/Yolov5m_Retrain_using_HCK_Full_GT/weights/best.pt \
--device 3 \
--name HCK_Val_Yolov5m_scratchGT_retrainGT_Conf01 \
--data data_HCK/HACKTHON_full.yaml \
--save-conf \
--save-txt \
--exist-ok



# Yolov5m - Retrain using goodGT failedPD
python val.py \
--weights runs/train/Shoaib/HCK/HCK_Exp4_Yolov5M_v1/Yolov5m_Retrain_using_HCK_passedGT_failedPD/weights/best.pt \
--device 3 \
--name HCK_Val_Yolov5m_scratchGT_retrainFiltered_Conf01 \
--data data_HCK/HACKTHON_full.yaml \
--save-conf \
--save-txt \
--exist-ok



# Yolov5m - Retrain using recovered goodGT failedPD
python val.py \
--weights runs/train/Shoaib/HCK/HCK_Exp4_Yolov5M_v2/Yolov5m_Retrain_using_HCK_passedGT_failedPD/weights/best.pt \
--device 3 \
--name HCK_Val_Yolov5m_scratchGT_recovered_retrainFiltered_Conf01 \
--data data_HCK/HACKTHON_full.yaml \
--save-conf \
--save-txt \
--exist-ok



 black blue green org red silver white yellow 


for val_data in 1color_with_aug_val
do
	for clr in black blue green org red silver white yellow
	do
		python val.py \
		--weights "runs/train/Aroona/HCK/Yolov5m_Retrain_using_HCK_segmented_augmented_starGANv2_10pIOU_org/weights/best.pt" \
		--device 0 \
		--name "Yolov5m_Retrain_using_HCK_segmented_augmented_starGANv2_10pIOU_org_vs_"$clr \
		--data "data_aroona/"$val_data"/HACKTHON_"$clr".yaml" \
		--exist-ok
	done
done



python train.py \
--batch-size 64 \
--data data_aroona/1color_with_aug_val/HACKTHON_org.yaml \
--cfg models/yolov5s.yaml \
--device 0 \
--name Aroona/HCK/Yolov5s/Yolov5s_Retrain_using_HCK_segmented_augmented_starGANv2_10pIOU_org \
--weights runs/train/Aroona/HCK/Yolov5s/Yolov5s_Retrain_using_HCK_segmented_augmented_starGANv2_10pIOU_org/weights/last.pt  \
--epochs 1000 \
--exist-ok --cache --resume --start_epoch 500


python train.py \
--batch-size 64 \
--data data_aroona/1color_with_aug_val/HACKTHON_white.yaml \
--cfg models/yolov5s.yaml \
--device 1 \
--name Aroona/HCK/Yolov5s/Yolov5s_Retrain_using_HCK_segmented_augmented_starGANv2_10pIOU_white \
--weights runs/train/Aroona/HCK/Yolov5s/Yolov5s_Retrain_using_HCK_segmented_augmented_starGANv2_10pIOU_white/weights/last.pt  \
--epochs 1000 \
--exist-ok --cache --resume --start_epoch 500


python train.py \
--batch-size 64 \
--data data_aroona/1color_with_aug_val/HACKTHON_black.yaml \
--cfg models/yolov5s.yaml \
--device 2 \
--name Aroona/HCK/Yolov5s/Yolov5s_Retrain_using_HCK_segmented_augmented_starGANv2_10pIOU_black \
--weights runs/train/Aroona/HCK/Yolov5s/Yolov5s_Retrain_using_HCK_segmented_augmented_starGANv2_10pIOU_black/weights/last.pt  \
--epochs 1000 \
--exist-ok --cache --resume --start_epoch 500


python train.py \
--batch-size 64 \
--data data_aroona/1color_with_aug_val/HACKTHON_red.yaml \
--cfg models/yolov5s.yaml \
--device 3 \
--name Aroona/HCK/Yolov5s/Yolov5s_Retrain_using_HCK_segmented_augmented_starGANv2_10pIOU_red \
--weights runs/train/Aroona/HCK/Yolov5s/Yolov5s_Retrain_using_HCK_segmented_augmented_starGANv2_10pIOU_red/weights/last.pt  \
--epochs 1000 \
--exist-ok --cache --resume --start_epoch 500


python train.py \
--batch-size 64 \
--data data_aroona/1color_with_aug_val/HACKTHON_silver.yaml \
--cfg models/yolov5s.yaml \
--device 4 \
--name Aroona/HCK/Yolov5s/Yolov5s_Retrain_using_HCK_segmented_augmented_starGANv2_10pIOU_silver \
--weights runs/train/Aroona/HCK/Yolov5s/Yolov5s_Retrain_using_HCK_segmented_augmented_starGANv2_10pIOU_silver/weights/last.pt  \
--epochs 1000 \
--exist-ok --cache --resume --start_epoch 500


python train.py \
--batch-size 64 \
--data data_aroona/1color_with_aug_val/HACKTHON_blue.yaml \
--cfg models/yolov5s.yaml \
--device 5 \
--name Aroona/HCK/Yolov5s/Yolov5s_Retrain_using_HCK_segmented_augmented_starGANv2_10pIOU_blue \
--weights runs/train/Aroona/HCK/Yolov5s/Yolov5s_Train_from_scratch_using_full_GT/weights/best.pt  \
--epochs 1000 \
--exist-ok --cache


python train.py \
--batch-size 64 \
--data data_aroona/1color_with_aug_val/HACKTHON_green.yaml \
--cfg models/yolov5s.yaml \
--device 6 \
--name Aroona/HCK/Yolov5s/Yolov5s_Retrain_using_HCK_segmented_augmented_starGANv2_10pIOU_green \
--weights runs/train/Aroona/HCK/Yolov5s/Yolov5s_Train_from_scratch_using_full_GT/weights/best.pt  \
--epochs 1000 \
--exist-ok --cache


python train.py \
--batch-size 64 \
--data data_aroona/1color_with_aug_val/HACKTHON_yellow.yaml \
--cfg models/yolov5s.yaml \
--device 7 \
--name Aroona/HCK/Yolov5s/Yolov5s_Retrain_using_HCK_segmented_augmented_starGANv2_10pIOU_yellow \
--weights runs/train/Aroona/HCK/Yolov5s/Yolov5s_Train_from_scratch_using_full_GT/weights/best.pt  \
--epochs 1000 \
--exist-ok --cache