python train.py --batch-size 64 --data data_bdd/bdd_with_GANAug_StarGANv2.yaml \
--cfg models/yolov5s.yaml \
--weights runs/train/Yolov5s_BDD_GanAug_StarGANv2/weights/last.pt  \
--device 0 \
--name Yolov5s_BDD_GanAug_StarGANv2  \
--epochs 1000 \
--exist-ok \
--resume

python train.py --batch-size 64 --data data_bdd/bdd_with_GANAug_StarGANv2.yaml \
--cfg models/yolov5m.yaml \
--weights runs/train/Yolov5m_BDD_GanAug_StarGANv2/weights/last.pt \
--device 1 \
--name Yolov5m_BDD_GanAug_StarGANv2  \
--epochs 1000 \
--exist-ok \
--resume

python train.py --batch-size 64 --data data_bdd/bdd_org.yaml \
--cfg models/yolov5s.yaml \
--weights runs/train/Yolov5s_BDD_Org/weights/last.pt  \
--device 2 \
--name Yolov5s_BDD_Org  \
--epochs 1000 \
--exist-ok \
--resume

python train.py --batch-size 64 --data data_bdd/bdd_org.yaml \
--cfg models/yolov5m.yaml \
--weights runs/train/Yolov5m_BDD_Org/weights/last.pt  \
--device 3 \
--name Yolov5m_BDD_Org  \
--epochs 1000 \
--exist-ok \
--resume

python train.py --batch-size 64 --data data_HCK/HACKTHON_full.yaml \
--cfg models/yolov5s.yaml \
--weights runs/train/HCK_Train_from_scratch/weights/last.pt  \
--device 4 \
--name HCK_Train_from_scratch  \
--epochs 1000 \
--exist-ok \
--cache \
--resume

python train.py --batch-size 64 --data data_HCK/HACKTHON_full_good_from_scratch_best.yaml \
--cfg models/yolov5s.yaml \
--weights runs/train/HCK_Train_from_pretrained_HCK_good/weights/last.pt  \
--device 5 \
--name HCK_Train_from_pretrained_HCK_good  \
--epochs 1000 \
--exist-ok \
--cache \
--resume

python train.py --batch-size 64 --data data_HCK/HACKTHON_full_org-good_pred-failed.yaml \
--cfg models/yolov5s.yaml \
--weights runs/train/HCK_Train_from_pretrained_HCK_GT-good_Pred-failed/weights/last.pt  \
--device 6 \
--name HCK_Train_from_pretrained_HCK_GT-good_Pred-failed  \
--epochs 1000 \
--exist-ok \
--cache \
--resume

python train.py --batch-size 64 --data data_HCK/HACKTHON_full.yaml \
--cfg models/yolov5s.yaml \
--weights runs/train/HCK_Train_from_pretrained_using_HCK_full/weights/last.pt  \
--device 7 \
--name HCK_Train_from_pretrained_using_HCK_full  \
--epochs 1000 \
--exist-ok \
--cache \
--resume





# Multi-GPU trainings


python train.py \
-m torch.distributed.run \
--nproc_per_node 2 \
--sync-bn \
--batch-size 64 \
--data data_HCK/HACKTHON_full.yaml \
--cfg models/yolov5s.yaml \
--weights runs/train/HCK_Train_from_scratch/weights/last.pt  \
--device 0,1 \
--name HCK_Train_from_scratch  \
--epochs 1000 \
--exist-ok \
--cache --resume

python train.py \
-m torch.distributed.run \
--nproc_per_node 2 \
--sync-bn \
--batch-size 64 \
--data data_HCK/HACKTHON_full_good_from_scratch_best.yaml \
--cfg models/yolov5s.yaml \
--weights runs/train/HCK_Train_from_pretrained_HCK_good/weights/last.pt  \
--device 2,3 \
--name HCK_Train_from_pretrained_HCK_good  \
--epochs 1000 \
--exist-ok \
--cache --resume

python train.py \
-m torch.distributed.run \
--nproc_per_node 2 \
--sync-bn \
--batch-size 64 \
--data data_HCK/HACKTHON_full_org-good_pred-failed.yaml \
--cfg models/yolov5s.yaml \
--weights runs/train/HCK_Train_from_pretrained_HCK_GT-good_Pred-failed/weights/last.pt  \
--device 4,5 \
--name HCK_Train_from_pretrained_HCK_GT-good_Pred-failed  \
--epochs 1000 \
--exist-ok \
--cache --resume

python train.py \
-m torch.distributed.run \
--nproc_per_node 2 \
--sync-bn \
--batch-size 64 \
--data data_HCK/HACKTHON_full.yaml \
--cfg models/yolov5s.yaml \
--weights runs/train/HCK_Train_from_pretrained_using_HCK_full/weights/last.pt  \
--device 6,7 \
--name HCK_Train_from_pretrained_using_HCK_full  \
--epochs 1000 \
--exist-ok \
--cache --resume




python val.py --weights runs/train/Yolov5m_BDD_Org/weights/best.pt --device 0 --name Val_Train_Yolov5m_BDD-Org_Val_BDD-Org --data data_bdd/bdd_org.yaml
python val.py --weights runs/train/Yolov5m_BDD_Org/weights/best.pt --device 0 --name Val_Train_Yolov5m_BDD-Org_Val_BDD-GAN --data data_bdd/bdd_with_GANAug_StarGANv2.yaml
python val.py --weights runs/train/Yolov5m_BDD_Org/weights/best.pt --device 0 --name Val_Train_Yolov5m_BDD-Org_Val_HCK-Org --data data_HCK/HACKTHON_full.yaml

python val.py --weights runs/train/Yolov5m_BDD_GanAug_StarGANv2/weights/best.pt --device 0 --name Val_Train_Yolov5m_BDD-GAN_Val_BDD-Org --data data_bdd/bdd_org.yaml
python val.py --weights runs/train/Yolov5m_BDD_GanAug_StarGANv2/weights/best.pt --device 0 --name Val_Train_Yolov5m_BDD-GAN_Val_BDD-GAN --data data_bdd/bdd_with_GANAug_StarGANv2.yaml
python val.py --weights runs/train/Yolov5m_BDD_GanAug_StarGANv2/weights/best.pt --device 0 --name Val_Train_Yolov5m_BDD-GAN_Val_HCK-Org --data data_HCK/HACKTHON_full.yaml

python val.py --weights runs/train/Yolov5s_BDD_Org/weights/best.pt --device 0 --name Val_Train_Yolov5s_BDD-Org_Val_BDD-Org --data data_bdd/bdd_org.yaml
python val.py --weights runs/train/Yolov5s_BDD_Org/weights/best.pt --device 0 --name Val_Train_Yolov5s_BDD-Org_Val_BDD-GAN --data data_bdd/bdd_with_GANAug_StarGANv2.yaml
python val.py --weights runs/train/Yolov5s_BDD_Org/weights/best.pt --device 0 --name Val_Train_Yolov5s_BDD-Org_Val_HCK-Org --data data_HCK/HACKTHON_full.yaml

python val.py --weights runs/train/Yolov5s_BDD_GanAug_StarGANv2/weights/best.pt --device 0 --name Val_Train_Yolov5s_BDD-GAN_Val_BDD-Org --data data_bdd/bdd_org.yaml
python val.py --weights runs/train/Yolov5s_BDD_GanAug_StarGANv2/weights/best.pt --device 0 --name Val_Train_Yolov5s_BDD-GAN_Val_BDD-GAN --data data_bdd/bdd_with_GANAug_StarGANv2.yaml
python val.py --weights runs/train/Yolov5s_BDD_GanAug_StarGANv2/weights/best.pt --device 0 --name Val_Train_Yolov5s_BDD-GAN_Val_HCK-Org --data data_HCK/HACKTHON_full.yaml




# BDD Validation
python val.py --weights /data/trained_weights/Exp4/Training_with_full_data_yolov5s/weights/best.pt --data 