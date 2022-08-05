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

