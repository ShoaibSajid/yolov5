# -m torch.distributed.launch --nproc_per_node 2 --master_port 0123

torchrun --nproc_per_node=2  --master_port=23 train.py \
--batch-size 64 \
--data /home/shoaib/codes/Yolov5_DeepSORT_PseudoLabels/yolov5/data_HCK/HACKTHON_full.yaml \
--cfg models/yolov5s.yaml \
--device 2,3 \            
--name runs/train/Shoaib/HCK/HCK_Exp5_Yolov5S_v2/Yolov5s_Retrain_using_HCK_Full_GT \
--weights runs/train/Shoaib/HCK/HCK_Exp5_Yolov5S_v2/Yolov5s_Train_from_scratch_using_full_GT/weights/best.pt  \ 
--epochs 500 \
--exist-ok --cache

torchrun --nproc_per_node=2  --master_port=23 train.py \
--batch-size 64 \
--data /home/shoaib/codes/Yolov5_DeepSORT_PseudoLabels/yolov5/data_HCK/HCK_recover_filter_yolov5s/HACKTHON_full_org-good_pred-failed.yaml \
--cfg models/yolov5s.yaml \
--device 2,3 \            
--name Shoaib/HCK/HCK_Exp5_Yolov5S_v2/Yolov5s_Retrain_using_passedGT_failedPD \
--weights runs/train/Shoaib/HCK/HCK_Exp5_Yolov5S_v2/Yolov5s_Train_from_scratch_using_full_GT/weights/best.pt  \ 
--epochs 500 \
--exist-ok --cache