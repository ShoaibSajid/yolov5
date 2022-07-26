#! Create psuedo labels
python detect.py --weights runs/train/BDD_Day_stage1_n10k/weights/last.pt 	--device 0,1 	--name BDD_Psuedo_stage2_train_labels  	--save-txt  --nosave  --source /data/datasets/BDD/BDD_Day_Night_Experiment/BDD_Night_Psuedo_labels_stage_2/train     

python detect.py --weights runs/train/BDD_Day_stage1_n10k/weights/last.pt 	--device 0,1 	--name BDD_Psuedo_stage2_val_labels 	--save-txt  --nosave  --source /data/datasets/BDD/BDD_Day_Night_Experiment/BDD_Night_Psuedo_labels_stage_2/val

python detect.py --weights runs/train/BDD_D 	--device 0,1 	--name BDD_Val_Cropped 	--save-txt --save-crop  --nosave  --source /data/datasets/BDD/BDD_Day_Night_Experiment/BDD_Night_Psuedo_labels_stage_2/val


#! Move pseudo labels to destination
mv runs/detect/BDD_Psuedo_stage2_train_labels/labels/* 	/data/datasets/BDD/BDD_Day_Night_Experiment/BDD_Night_Psuedo_labels_stage_2/train/
mv runs/detect/BDD_Psuedo_stage2_val_labels/labels/* 	/data/datasets/BDD/BDD_Day_Night_Experiment/BDD_Night_Psuedo_labels_stage_2/val/

#! Run next training sequence
python -m torch.distributed.launch --nproc_per_node 2  train.py --batch-size 64 --data data/bdd_day_all_night_15k_stage2.yaml --cfg models/yolov5s.yaml --weights runs/train/BDD_Day_stage1_n10k/weights/last.pt --device 0,1 --name BDD_Day_stage2_n15k --epoch 400 --start_epoch 300

cp -r BDD_Night_Psuedo_labels_stage_0/train BDD_Night_Psuedo_labels_stage_1 
cp -r BDD_Night_Psuedo_labels_stage_0/val BDD_Night_Psuedo_labels_stage_1 