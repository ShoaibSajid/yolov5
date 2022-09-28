python train.py \
--batch-size 64 \
--data data_aroona/HACKTHON_augmented_starGANv2_10pIOU.yaml \
--cfg models/yolov5m.yaml \
--device 1 \
--name Aroona/HCK/Yolov5m_Retrain_using_HCK_augmented_starGANv2_10pIOU \
--weights runs/train/Aroona/HCK/Yolov5m_Train_from_scratch_using_full_GT/weights/best.pt  \
--epochs 500 \
--exist-ok --cache


python train.py \
--batch-size 64 \
--data data_aroona/HACKTHON_segmented_augmented_starGANv2_10pIOU.yaml \
--cfg models/yolov5m.yaml \
--device 5 \
--name Aroona/HCK/Yolov5m_Retrain_using_HCK_segmented_augmented_starGANv2_10pIOU \
--weights runs/train/Aroona/HCK/Yolov5m_Train_from_scratch_using_full_GT/weights/best.pt  \
--epochs 500 \
--exist-ok --cache


python train.py \
--batch-size 64 \
--data data_aroona/HACKTHON_original_extended.yaml \
--cfg models/yolov5m.yaml \
--device 0 \
--name Aroona/HCK/Yolov5m_Retrain_using_HCK_extended \
--weights runs/train/Aroona/HCK/Yolov5m_Train_from_scratch_using_full_GT/weights/best.pt  \
--epochs 500 \
--exist-ok --cache



/home/shoaib/codes/Yolov5_DeepSORT_PseudoLabels/yolov5/data_aroona/HACKTHON_original_extended.yaml