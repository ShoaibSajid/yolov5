
# Yolov5s

# Yolov5s - GT from scratch
python val.py \
--weights runs/train/Shoaib/HCK/HCK_Exp3_Org_Filtered/HCK_Train_from_scratch/weights/best.pt \
--device 2 \
--name HCK_Val_Yolov5s_scratchGT_Conf50 \
--data data_HCK/HACKTHON_full.yaml \
--conf 0.5 \
--save-conf \
--save-txt \
--exist-ok

# Yolov5s - Retrain using full
python val.py \
--weights runs/train/Shoaib/HCK/HCK_Exp3_Org_Filtered/HCK_Train_from_pretrained_using_HCK_full/weights/best.pt \
--device 2 \
--name HCK_Val_Yolov5s_scratchGT_retrainGT_Conf50 \
--data data_HCK/HACKTHON_full.yaml \
--conf 0.5 \
--save-conf \
--save-txt \
--exist-ok



# Yolov5s - Retrain using passedGT failedPD
python val.py \
--weights runs/train/Shoaib/HCK/HCK_Exp3_Org_Filtered/HCK_Train_from_pretrained_HCK_GT-good_Pred-failed/weights/best.pt \
--device 2 \
--name HCK_Val_Yolov5s_scratchGT_retrainFiltered_Conf50 \
--data data_HCK/HACKTHON_full.yaml \
--conf 0.5 \
--save-conf \
--save-txt \
--exist-ok



# Yolov5s - Retrain using recovered passedGT failedPD
python val.py \
--weights runs/train/Shoaib/HCK/HCK_Exp5_Yolov5S_v2/Yolov5s_Retrain_using_passedGT_failedPD/weights/best.pt \
--device 2 \
--name HCK_Val_Yolov5s_scratchGT_recovered_retrainFiltered_Conf50 \
--data data_HCK/HACKTHON_full.yaml \
--conf 0.5 \
--save-conf \
--save-txt \
--exist-ok


























# Yolov5s

# Yolov5s - GT from scratch
python val.py \
--weights runs/train/Shoaib/HCK/HCK_Exp3_Org_Filtered/HCK_Train_from_scratch/weights/best.pt \
--device 2 \
--name HCK_Val_Yolov5s_scratchGT_Conf01 \
--data data_HCK/HACKTHON_full.yaml \
--save-conf \
--save-txt \
--exist-ok

# Yolov5s - Retrain using full
python val.py \
--weights runs/train/Shoaib/HCK/HCK_Exp3_Org_Filtered/HCK_Train_from_pretrained_using_HCK_full/weights/best.pt \
--device 2 \
--name HCK_Val_Yolov5s_scratchGT_retrainGT_Conf01 \
--data data_HCK/HACKTHON_full.yaml \
--save-conf \
--save-txt \
--exist-ok



# Yolov5s - Retrain using passedGT failedPD
python val.py \
--weights runs/train/Shoaib/HCK/HCK_Exp3_Org_Filtered/HCK_Train_from_pretrained_HCK_GT-good_Pred-failed/weights/best.pt \
--device 2 \
--name HCK_Val_Yolov5s_scratchGT_retrainFiltered_Conf01 \
--data data_HCK/HACKTHON_full.yaml \
--save-conf \
--save-txt \
--exist-ok



# Yolov5s - Retrain using recovered passedGT failedPD
python val.py \
--weights runs/train/Shoaib/HCK/HCK_Exp5_Yolov5S_v2/Yolov5s_Retrain_using_passedGT_failedPD/weights/best.pt \
--device 2 \
--name HCK_Val_Yolov5s_scratchGT_recovered_retrainFiltered_Conf01 \
--data data_HCK/HACKTHON_full.yaml \
--save-conf \
--save-txt \
--exist-ok



