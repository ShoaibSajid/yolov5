# -m torch.distributed.launch --nproc_per_node 2 --master_port 0123


torchrun --nproc_per_node=4  --master_port=4567 train.py \           
--batch-size 64 \
--data /home/shoaib/codes/Yolov5_DeepSORT_PseudoLabels/yolov5/data_HCK/HACKTHON_full.yaml \
--cfg models/yolov5m.yaml \
--device 4,5,6,7 \        
--name runs/train/Shoaib/HCK/HCK_Exp4_Yolov5M_v2/Yolov5m_Retrain_using_HCK_Full_GT \
--weights runs/train/Shoaib/HCK/HCK_Exp4_Yolov5M_v2/Yolov5m_Train_from_scratch_using_full_GT/weights/best.pt  \ 
--epochs 500 \
--exist-ok --cache


torchrun --nproc_per_node=4  --master_port=4567 train.py \           
--batch-size 64 \
--data /home/shoaib/codes/Yolov5_DeepSORT_PseudoLabels/yolov5/data_HCK/HCK_filter_yolov5m/HACKTHON_full_org-good_pred-failed.yaml \
--cfg models/yolov5m.yaml \
--device 4,5,6,7 \        
--name Shoaib/HCK/HCK_Exp4_Yolov5M_v1/Yolov5m_Retrain_using_HCK_passedGT_failedPD \
--weights runs/train/Shoaib/HCK/HCK_Exp4_Yolov5M_v2/Yolov5m_Retrain_using_HCK_Full_GT/weights/last.pt  \ 
--epochs 500 \
--exist-ok --cache --resume

torchrun --nproc_per_node=4  --master_port=4567 train.py \
--batch-size 64 \
--data /home/shoaib/codes/Yolov5_DeepSORT_PseudoLabels/yolov5/data_HCK/HCK_filter_yolov5m/HACKTHON_full_org-good_pred-failed.yaml \
--cfg models/yolov5m.yaml \
--device 4,5,6,7 \
--name Shoaib/HCK/HCK_Exp4_Yolov5M_v1/Yolov5m_Retrain_using_HCK_passedGT_failedPD \
--weights runs/train/Shoaib/HCK/HCK_Exp4_Yolov5M_v2/Yolov5m_Train_from_scratch_using_full_GT/weights/best.pt  \
--epochs 500 \
--exist-ok --cache --resume