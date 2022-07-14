# Current Trainings

# Using BDD Full - Retrain from 0
python train.py \
--batch-size 64 \
--data data_bdd/bdd_org.yaml \
--cfg models/yolov5s.yaml \
--weights runs/train/Shoaib/BDD/BDD_Exp4_FullData_FilteredData_Start_0_Retry/Start_0/BDD_Train_FullBDD70k_Retrain_on_BDD70k_Org/weights/last.pt \
--device 7 \
--name Shoaib/BDD/BDD_Exp4_FullData_FilteredData_Start_0_Retry/Start_0/BDD_Train_FullBDD70k_Retrain_on_BDD70k_Org \
--resume \
--epochs 500 \
--exist-ok 

# Using BDD Full - Retrain from 300
python train.py \
--batch-size 64 \
--data data_bdd/bdd_org.yaml \
--cfg models/yolov5s.yaml \
--weights runs/train/Shoaib/BDD/BDD_Exp4_FullData_FilteredData_Start_0_Retry/Start_300/BDD_Train_FullBDD70k_Retrain_on_BDD70k_Org/weights/last.pt \
--device 6 \
--name Shoaib/BDD/BDD_Exp4_FullData_FilteredData_Start_0_Retry/Start_300/BDD_Train_FullBDD70k_Retrain_on_BDD70k_Org \
--resume \
--epochs 500 \
--exist-ok


# ------------------------------------

# Using BDD GoodGT FailedPD - Retrain from 0
python train.py \
--batch-size 64 \
--data data_bdd/bdd_org_goodGT_failedPD.yaml \
--cfg models/yolov5s.yaml \
--weights runs/train/Shoaib/BDD/BDD_Exp4_FullData_FilteredData_Start_0_Retry/Start_0/BDD_Train_FullBDD70k_Retrain_on_bdd_goodGT_failedPD/weights/last.pt \
--device 5 \
--name Shoaib/BDD/BDD_Exp4_FullData_FilteredData_Start_0_Retry/Start_0/BDD_Train_FullBDD70k_Retrain_on_bdd_goodGT_failedPD \
--resume \
--epochs 500 \
--exist-ok \
--cache

# Using BDD GoodGT FailedPD - Retrain from 300
python train.py \
--batch-size 64 \
--data data_bdd/bdd_org_goodGT_failedPD.yaml \
--cfg models/yolov5s.yaml \
--weights runs/train/Shoaib/BDD/BDD_Exp4_FullData_FilteredData_Start_0_Retry/Start_300/BDD_Train_FullBDD70k_Retrain_on_bdd_goodGT_failedPD/weights/last.pt \
--device 4 \
--name Shoaib/BDD/BDD_Exp4_FullData_FilteredData_Start_0_Retry/Start_300/BDD_Train_FullBDD70k_Retrain_on_bdd_goodGT_failedPD \
--resume \
--epochs 500 \
--exist-ok \
--cache


# ------------------------------------

# Using BDD Good only - Retrain from 0
python train.py \
--batch-size 64 \
--data data_bdd/bdd_org_goodGT.yaml \
--cfg models/yolov5s.yaml \
--weights runs/train/Shoaib/BDD/BDD_Exp4_FullData_FilteredData_Start_0_Retry/Start_0/BDD_Train_FullBDD70k_Retrain_on_bdd_goodGT/weights/last.pt \
--device 3 \
--name Shoaib/BDD/BDD_Exp4_FullData_FilteredData_Start_0_Retry/Start_0/BDD_Train_FullBDD70k_Retrain_on_bdd_goodGT \
--resume \
--epochs 500 \
--exist-ok \
--cache

# Using BDD Good only - Retrain from 300
python train.py \
--batch-size 64 \
--data data_bdd/bdd_org_goodGT.yaml \
--cfg models/yolov5s.yaml \
--weights runs/train/Shoaib/BDD/BDD_Exp4_FullData_FilteredData_Start_0_Retry/Start_300/BDD_Train_FullBDD70k_Retrain_on_bdd_goodGT/weights/last.pt \
--device 2 \
--name Shoaib/BDD/BDD_Exp4_FullData_FilteredData_Start_0_Retry/Start_300/BDD_Train_FullBDD70k_Retrain_on_bdd_goodGT \
--resume \
--epochs 500 \
--exist-ok \
--cache 


# ------------------------------------

# StarGANv2_Seg
python train.py \
--batch-size 64 \
--data data_bdd/bdd_with_GANAug_StarGANv2.yaml \
--cfg models/yolov5s.yaml \
--weights runs/train/Aroona/Yolov5s_BDD_GanAug_StarGANv2_Seg/weights/last.pt \
--device 1 \
--name Aroona/Yolov5s_BDD_GanAug_StarGANv2_Seg \
--epochs 1000 \
--resume \
--exist-ok
#--cache 


python train.py \
--batch-size 64 \
--data data_bdd/bdd_with_GANAug_StarGANv2.yaml \
--cfg models/yolov5m.yaml \
--weights runs/train/Aroona/Yolov5m_BDD_GanAug_StarGANv2_Seg/weights/last.pt \
--device 0 \
--name Aroona/Yolov5m_BDD_GanAug_StarGANv2_Seg \
--epochs 1000 \
--resume \
--exist-ok

# ------------------------------------



# python -m torch.distributed.launch --nproc_per_node 2 --master_port 0123 train.py --sync-bn --batch-size 64 --data data_bdd/bdd_org_goodGT.yaml --cfg models/yolov5s.yaml --weights ./runs/train/Shoaib/BDD/BDD_Exp1_FullData_FilteredData_Start_300/BDD_Training_with_full_data_yolov5s_from_scratch/weights/best.pt --device 6,7 --name Shoaib/BDD/BDD_Exp3_FullData_FilteredData_Start_0/BDD_Train_on_DayNight_Retrain_on_BDD70k_Only_Good50k --epochs 500 --exist-ok --cache --resume




python -m torch.distributed.launch --nproc_per_node 4 --master_port 0123 train.py \
--batch-size 64 \
--data data_mixed/mixed_train.txt \
--cfg models/yolov5s.yaml \
--weights '' \
--device 4,5,6,7 \
--name Shoaib/BDD-HCK/Start_from_scratch \
--epochs 500 \
--resume \
--exist-ok








python -m torch.distributed.launch --nproc_per_node 2 --master_port 0123 train.py \
--batch-size 64 \
--data data_bdd/bdd_with_GANAug_StarGANv2_seg_additional_pseudos.yaml \
--cfg models/yolov5s.yaml \
--weights '' \
--device 0,1 \
--name Shoaib/BDD/BDD_Exp5_FullData_and_FailedData \
--epochs 500 \
--resume \
--exist-ok