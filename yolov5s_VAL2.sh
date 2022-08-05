

# Yolov5s - GT from scratch
python val.py \
--weights runs/train/Shoaib/HCK/HCK_Exp5_Yolov5S_v2/Yolov5s_Train_from_scratch_using_full_GT/weights/best.pt \
--device 1 \
--name HCK_Val_Yolov5s_scratchGT_2_Conf50_Exp5 \
--data data_HCK/HACKTHON_full.yaml \
--conf 0.5 \
--save-conf \
--save-txt \
--exist-ok

# Yolov5s - Retrain using full
python val.py \
--weights runs/train/Shoaib/HCK/HCK_Exp5_Yolov5S_v2/Yolov5s_Retrain_using_HCK_Full_GT/weights/best.pt \
--device 1 \
--name HCK_Val_Yolov5s_scratchGT_retrainGT_2_Conf50_Exp5 \
--data data_HCK/HACKTHON_full.yaml \
--conf 0.5 \
--save-conf \
--save-txt \
--exist-ok







# Yolov5s - Retrain using passedGT failedPD
python val.py \
--weights runs/train/Shoaib/HCK/HCK_Exp3_Org_Filtered/HCK_Train_from_pretrained_HCK_GT-good_Pred-failed/weights/best.pt \
--device 1 \
--name HCK_Val_Yolov5s_scratchGT_retrainFiltered_Conf50_Exp5 \
--data data_HCK/HACKTHON_full.yaml \
--conf 0.5 \
--save-conf \
--save-txt \
--exist-ok



# Yolov5s - Retrain using recovered passedGT failedPD
python val.py \
--weights runs/train/Shoaib/HCK/HCK_Exp5_Yolov5S_v2/Yolov5s_Retrain_using_passedGT_failedPD/weights/best.pt \
--device 1 \
--name HCK_Val_Yolov5s_scratchGT_recovered_retrainFiltered_Conf50_Exp5 \
--data data_HCK/HACKTHON_full.yaml \
--conf 0.5 \
--save-conf \
--save-txt \
--exist-ok


















# Yolov5s - GT from scratch
python val.py \
--weights runs/train/Shoaib/HCK/HCK_Exp5_Yolov5S_v2/Yolov5s_Train_from_scratch_using_full_GT/weights/best.pt \
--device 1 \
--name HCK_Val_Yolov5s_scratchGT_2_Conf01_Exp5 \
--data data_HCK/HACKTHON_full.yaml \
--save-conf \
--save-txt \
--exist-ok

# Yolov5s - Retrain using full
python val.py \
--weights runs/train/Shoaib/HCK/HCK_Exp5_Yolov5S_v2/Yolov5s_Retrain_using_HCK_Full_GT/weights/best.pt \
--device 1 \
--name HCK_Val_Yolov5s_scratchGT_retrainGT_2_Conf01_Exp5 \
--data data_HCK/HACKTHON_full.yaml \
--save-conf \
--save-txt \
--exist-ok







# Yolov5s - Retrain using passedGT failedPD
python val.py \
--weights runs/train/Shoaib/HCK/HCK_Exp3_Org_Filtered/HCK_Train_from_pretrained_HCK_GT-good_Pred-failed/weights/best.pt \
--device 1 \
--name HCK_Val_Yolov5s_scratchGT_retrainFiltered_Conf01_Exp5 \
--data data_HCK/HACKTHON_full.yaml \
--save-conf \
--save-txt \
--exist-ok



# Yolov5s - Retrain using recovered passedGT failedPD
python val.py \
--weights runs/train/Shoaib/HCK/HCK_Exp5_Yolov5S_v2/Yolov5s_Retrain_using_passedGT_failedPD/weights/best.pt \
--device 1 \
--name HCK_Val_Yolov5s_scratchGT_recovered_retrainFiltered_Conf01_Exp5 \
--data data_HCK/HACKTHON_full.yaml \
--save-conf \
--save-txt \
--exist-ok


