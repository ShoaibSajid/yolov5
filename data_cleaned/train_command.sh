python train.py \
--batch-size 64 \
--data data_cleaned/1_HACKTHON_cleaned_GT.yaml \
--cfg models/yolov5m.yaml \
--device 5 \
--name Shoaib/HCK/HCK_Exp6_Cleaned/yolov5m \
--weights runs/train/Shoaib/HCK/HCK_Exp6_Cleaned/yolov5m/Train_with_cleaned_GT/weights/last.pt  \
--epochs 500 \
--exist-ok --cache --resume


python train.py \
--batch-size 64 \
--data data_cleaned/1_bdd_cleaned_GT.yaml \
--cfg models/yolov5m.yaml \
--device 4 \
--name Shoaib/BDD/BDD_Exp9_Cleaned/yolov5m/Train_with_cleaned_GT \
--weights ""  \
--epochs 500 \
--exist-ok --cache --resume

#YOLOV5m

##########################

#YOLOV5s

python train.py \
--batch-size 64 \
--data data_cleaned/1_bdd_cleaned_GT.yaml \
--cfg models/yolov5s.yaml \
--device 4 \
--name Shoaib/BDD/BDD_Exp9_Cleaned/yolov5s/Train_with_cleaned_GT \
--weights runs/train/Shoaib/BDD/BDD_Exp9_Cleaned/yolov5s/Train_with_cleaned_GT/weights/last.pt \
--epochs 500 \
--exist-ok --resume --cache



python train.py --batch-size 64 --data data_cleaned/1_HACKTHON_cleaned_GT.yaml --cfg models/yolov5s.yaml --device 7 --name Shoaib/HCK/HCK_Exp6_Cleaned/yolov5s/Train_with_cleaned_GT --weights " "  --epochs 500 --exist-ok


python train.py \
--batch-size 64 \
--data data_aroona/1color_with_aug_val/HACKTHON_all_colors.yaml  \
--cfg models/yolov5s.yaml \
--device 3 \
--name Aroona/HCK/Yolov5s/Yolov5s_Retrain_using_HCK_segmented_augmented_starGANv2_10pIOU_all_colors \
--weights runs/train/Aroona/HCK/Yolov5s/Yolov5s_Retrain_using_HCK_segmented_augmented_starGANv2_10pIOU_all_colors/weights/last.pt  \
--epochs 500 \
--exist-ok --resume


python train.py \
--batch-size 64 \
--data data_aroona/1color_with_aug_val/HACKTHON_all_colors.yaml \
--cfg models/yolov5m.yaml \
--device 1,2 \
--name Aroona/HCK/Yolov5m/Yolov5m_Retrain_using_HCK_segmented_augmented_starGANv2_10pIOU_all_colors \
--weights ./runs/train/Aroona/HCK/Yolov5m/Yolov5m_Retrain_using_HCK_segmented_augmented_starGANv2_10pIOU_all_colors/weights/last.pt  \
--epochs 500 \
--exist-ok --resume


python trackv2_from_file.py \
--source /data/pseudos/HCK/Exp6_Yolov5s/Yolov5s_HCK_TrainingData_CleanedWeights_Recover_Filter_Clean-0.001 \
--device 6 \
--output /data/pseudos/HCK/Exp6_Yolov5s/Yolov5s_HCK_TrainingData_CleanedWeights_Recover_Filter_Clean-0.5-FW \
--conf-thres 0.5 \
--iou-thres 0.6 \
--save-txt \
--exist-ok \
--deep_sort_model resnet50_MSMT17 \
--yolo_model /home/shoaib/codes/Yolov5_DeepSORT_PseudoLabels/yolov5/runs/train/Shoaib/HCK/HCK_Exp6_Cleaned/yolov5s/Train_with_cleaned_GT/weights/best.pt






######------------------ Retrain 3 HCK - YOLO V5 Small ---------------------



python train.py \
--batch-size 64 \
--data data_cleaned/4_HACKTHON_cleaned_GT.yaml \
--cfg models/yolov5s.yaml \
--device 6 \
--name Shoaib/HCK/HCK_Exp6_Cleaned/yolov5s/ReTrain_with_cleaned_GT \
--weights runs/train/Shoaib/HCK/HCK_Exp6_Cleaned/yolov5s/Train_with_cleaned_GT/weights/best.pt \
--epochs 500 \
--exist-ok --cache --resume


python train.py \
--batch-size 64 \
--data data_cleaned/3_HACKTHON_passedGT_failedPD.yaml \
--cfg models/yolov5s.yaml \
--device 4 \
--name Shoaib/HCK/HCK_Exp6_Cleaned/yolov5s/ReTrain_passedGT_failedPD \
--weights runs/train/Shoaib/HCK/HCK_Exp6_Cleaned/yolov5s/ReTrain_passedGT_failedPD/weights/last.pt \
--epochs 500 \
--exist-ok --cache --resume




######------------------ Retrain 3 BDD - YOLO V5 Small  ---------------------



python train.py \
--batch-size 64 \
--data data_cleaned/4_HACKTHON_cleaned_GT.yaml \
--cfg models/yolov5s.yaml \
--device 6 \
--name Shoaib/BDD/BDD_Exp9_Cleaned/yolov5s/ReTrain_with_cleaned_GT \
--weights runs/train/Shoaib/BDD/BDD_Exp9_Cleaned/yolov5s/Train_with_cleaned_GT/weights/last.pt \
--epochs 500 \
--exist-ok --cache --resume


python train.py \
--batch-size 64 \
--data data_cleaned/3_bdd_passedGT_failedPD.yaml \
--cfg models/yolov5s.yaml \
--device 4 \
--name Shoaib/BDD/BDD_Exp9_Cleaned/yolov5s/ReTrain_passedGT_failedPD \
--weights runs/train/Shoaib/BDD/BDD_Exp9_Cleaned/yolov5s/ReTrain_passedGT_failedPD/weights/last.pt \
--epochs 500 \
--exist-ok --cache --resume


######------------------ Retrain 3 HCK - YOLO V5 Mediu ---------------------



python train.py \
--batch-size 64 \
--data data_cleaned/3_HACKTHON_passedGT_failedPD.yaml \
--cfg models/yolov5m.yaml \
--device 4 \
--name Shoaib/HCK/HCK_Exp6_Cleaned/yolov5s/ReTrain_passedGT_failedPD \
--weights "" \
--epochs 500 \
--exist-ok --cache --resume


python train.py \
--batch-size 64 \
--data data_cleaned/4_HACKTHON_cleaned_GT.yaml \
--cfg models/yolov5m.yaml \
--device 6 \
--name Shoaib/HCK/HCK_Exp6_Cleaned/yolov5m/ReTrain_with_cleaned_GT \
--weights runs/train/Shoaib/HCK/HCK_Exp6_Cleaned/yolov5m/Train_with_cleaned_GT/weights/best.pt \
--epochs 500 \
--exist-ok --cache --resume




######------------------ Retrain 3 BDD - YOLO V5 Mediu  ---------------------



python train.py \
--batch-size 64 \
--data data_cleaned/3_bdd_passedGT_failedPD.yaml \
--cfg models/yolov5s.yaml \
--device 4 \
--name Shoaib/BDD/BDD_Exp9_Cleaned/yolov5s/ReTrain_passedGT_failedPD \
--weights "" \
--epochs 500 \
--exist-ok --cache --resume


python train.py \
--batch-size 64 \
--data data_cleaned/3_bdd_passedGT_failedPD-Medium.yaml \
--cfg models/yolov5m.yaml \
--device 4,5 \
--name Shoaib/BDD/BDD_Exp9_Cleaned/yolov5m/ReTrain_passedGT_failedPD \
--weights runs/train/Shoaib/BDD/BDD_Exp9_Cleaned/yolov5m/ReTrain_passedGT_failedPD/weights/last.pt \
--epochs 500 \
--exist-ok --cache --resume


python train.py --batch-size 64 --data data_cleaned/4_bdd_cleaned_GT.yaml --cfg models/yolov5m.yaml --device 0,1 --name Shoaib/BDD/BDD_Exp9_Cleaned/yolov5m/ReTrain_with_cleaned_GT --weights runs/train/Shoaib/BDD/BDD_Exp9_Cleaned/yolov5m/Train_with_cleaned_GT/weights/best.pt --epochs 500 --exist-ok --cache


python train.py --batch-size 64 --data data_cleaned/3_HACKTHON_passedGT_failedPD.yaml --cfg models/yolov5s.yaml --device 6 --name Shoaib/HCK/HCK_Exp6_Cleaned/yolov5s/Evolve_passedGT_failedPD --weights runs/train/Shoaib/HCK/HCK_Exp6_Cleaned/yolov5s/Train_with_cleaned_GT/weights/best.pt --epochs 100 --exist-ok --cache --evolve 











































##################### VAL ##############################

##############
# 1 - Org

# cd yolov5; clear;
# python val.py \
# --weights runs/train/Shoaib/HCK/HCK_Exp6_Cleaned/yolov5s/ReTrain_passedGT_failedPD/weights/best.pt \
# --device 7 \
# --name HCK_Exp6_Yolov5s_Retraining_Val_P-GT_F-PD \
# --data data_cleaned/5_HACKTHON_cleaned_GT.yaml \
# --exist-ok


# cd yolov5; clear;
# python val.py \
# --weights runs/train/Shoaib/HCK/HCK_Exp6_Cleaned/yolov5s/ReTrain_with_cleaned_GT/weights/best.pt \
# --device 7 \
# --name HCK_Exp6_Yolov5s_Retraining_Val \
# --data data_cleaned/5_HACKTHON_cleaned_GT.yaml \
# --exist-ok

##############
# 2 - Cleaned Pred

# cd yolov5; clear;
# python val.py \
# --weights runs/train/Shoaib/HCK/HCK_Exp6_Cleaned/yolov5m/ReTrain_passedGT_failedPD/weights/best.pt \
# --device 7 \
# --name HCK_Exp6_Yolov5s_Retraining_Val_P-GT_F-PD_cleaned_recleaned-GT \
# --data data_cleaned/5_HACKTHON_cleaned_GT.yaml \
# --exist-ok --clean_labels


# cd yolov5; clear;
# python val.py \
# --weights runs/train/Shoaib/HCK/HCK_Exp6_Cleaned/yolov5m/ReTrain_with_cleaned_GT/weights/best.pt \
# --device 7 \
# --name HCK_Exp6_Yolov5s_Retraining_Val_cleaned_recleaned-GT \
# --data data_cleaned/5_HACKTHON_cleaned_GT.yaml \
# --exist-ok --clean_labels

############
# 3 - Cleaned GT & PD

# cd yolov5; clear;
# python val.py \
# --weights runs/train/Shoaib/HCK/HCK_Exp6_Cleaned/yolov5m/ReTrain_passedGT_failedPD/weights/best.pt \
# --device 7 \
# --name HCK_Exp6_Yolov5s_Retraining_Val_P-GT_F-PD_cleaned_recleanGT \
# --data data_cleaned/5_HACKTHON_cleaned_GT.yaml \
# --exist-ok --clean_labels


# cd yolov5; clear;
# python val.py \
# --weights runs/train/Shoaib/HCK/HCK_Exp6_Cleaned/yolov5m/ReTrain_with_cleaned_GT/weights/best.pt \
# --device 7 \
# --name HCK_Exp6_Yolov5s_Retraining_Val_cleaned_recleanGT \
# --data data_cleaned/5_HACKTHON_cleaned_GT.yaml \
# --exist-ok --clean_labels


# ################## VAL - BDD #########################

# ##############
# # 1 - Orgd

# cd yolov5; clear;
# python val.py \
# --weights runs/train/Shoaib/BDD/BDD_Exp9_Cleaned/yolov5s/ReTrain_passedGT_failedPD/weights/best.pt \
# --device 7 \
# --name BDD_Exp9_Yolov5s_Retraining_Val_P-GT_F-PD \
# --data data_cleaned/5_bdd_cleaned_GT.yaml \
# --exist-ok


# cd yolov5; clear;
# python val.py \
# --weights runs/train/Shoaib/BDD/BDD_Exp9_Cleaned/yolov5s/ReTrain_with_cleaned_GT/weights/best.pt \
# --device 7 \
# --name BDD_Exp9_Yolov5s_Retraining_Val \
# --data data_cleaned/5_bdd_cleaned_GT.yaml \
# --exist-ok

# ##############
# # 2 - Cleaned Pred

# cd yolov5; clear;
# python val.py \
# --weights runs/train/Shoaib/BDD/BDD_Exp9_Cleaned/yolov5s/ReTrain_passedGT_failedPD/weights/best.pt \
# --device 4 \
# --name BDD_Exp9_Yolov5s_Retraining_Val_P-GT_F-PD_cleaned \
# --data data_cleaned/5_bdd_cleaned_GT.yaml \
# --exist-ok --clean_labels


# cd yolov5; clear;
# python val.py \
# --weights runs/train/Shoaib/BDD/BDD_Exp9_Cleaned/yolov5s/ReTrain_with_cleaned_GT/weights/best.pt \
# --device 4 \
# --name BDD_Exp9_Yolov5s_Retraining_Val_cleaned \
# --data data_cleaned/5_bdd_cleaned_GT.yaml \
# --exist-ok --clean_labels

# ############
# # 3 - Cleaned GT & PD

cd yolov5; clear;
python val.py \
--weights runs/train/Shoaib/BDD/BDD_Exp9_Cleaned/yolov5m/ReTrain_passedGT_failedPD/weights/best.pt \
--device 7 \
--name BDD_Exp9_Yolov5m_Retraining_Val_P-GT_F-PD_cleaned_recleanGT \
--data data_cleaned/5_bdd_cleaned_GT.yaml \
--exist-ok --clean_labels


cd yolov5; clear;
python val.py \
--weights runs/train/Shoaib/BDD/BDD_Exp9_Cleaned/yolov5m/ReTrain_with_cleaned_GT/weights/best.pt \
--device 4 \
--name BDD_Exp9_Yolov5m_Retraining_Val_cleaned_recleanGT \
--data data_cleaned/5_bdd_cleaned_GT.yaml \
--exist-ok --clean_labels





# python train.py --batch-size 64 --data data_cleaned/3_HACKTHON_passedGT_failedPD-Medium.yaml --cfg models/yolov5m.yaml --device 4 --name Shoaib/HCK/HCK_Exp6_Cleaned/yolov5m/ReTrain_passedGT_failedPD --weights runs/train/Shoaib/HCK/HCK_Exp6_Cleaned/yolov5m/Train_with_cleaned_GT/weights/best.pt --epochs 500 --exist-ok --cache




# echo " "
# for val_data in 1color_with_aug_val                
# do                
#         for clr in org black white blue red silver yellow green 
#         do
#                 echo " "
#                 echo " "
#                 echo " "
#                 echo " "
#                 echo "Val for "$clr" with augmented validation."
#                 echo " "
#                 python val.py \
#                 --weights "runs/train/Aroona/HCK/Yolov5s/Yolov5s_Retrain_using_HCK_segmented_augmented_starGANv2_10pIOU_"$clr"/weights/best.pt" \
#                 --device 5 \
#                 --name "Yolov5s_Retrain_using_HCK_segmented_augmented_starGANv2_10pIOU_"$clr \
#                 --data "data_aroona/"$val_data"/HACKTHON_"$clr".yaml" \
#                 --exist-ok
#         done
# done




# echo " "
# for val_data in 1color              
# do                
#         for clr in org black white blue red silver yellow green 
#         do
#                 echo " "
#                 echo " "
#                 echo " "
#                 echo " "
#                 echo "Val for "$clr" with original validation."
#                 echo " "
#                 python val.py \
#                 --weights "runs/train/Aroona/HCK/Yolov5m/Yolov5m_Retrain_using_HCK_segmented_augmented_starGANv2_10pIOU_"$clr"/weights/best.pt" \
#                 --device 1 \
#                 --name "Yolov5m_Retrain_using_HCK_segmented_augmented_starGANv2_10pIOU_org_val_"$clr \
#                 --data "data_aroona/"$val_data"/HACKTHON_"$clr".yaml" \
#                 --exist-ok
#         done
# done

