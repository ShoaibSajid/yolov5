cd ../../

root=/home/shoaib/codes/Yolov5_DeepSORT_PseudoLabels
YOLO=$root/yolov5

NAME="Yolov5m_BDD_TrainingData_CleanedWeights_Recover_Filter_Clean"
Yolo_weights=$YOLO/runs/train/Shoaib/BDD/BDD_Exp9_Cleaned/yolov5m/Train_with_cleaned_GT/weights/best.pt
DATASET="BDD"
EXP="Exp9_Yolov5m"

DS="resnet50_MSMT17"

LC="0.001"
HC="0.5"
IoU="0.6"
Batch="64"
Device="7"
DATA=$YOLO/data_cleaned/2_bdd_cleaned_GT_valistrain.yaml
DATA_GT="/home/shoaib/datasets/BDD/BDD_100k_Cleaned_g20_ar3/train"
DATA_list="/home/shoaib/datasets/BDD/BDD_100k_Cleaned_g20_ar3/train_list.txt"

LC_Labels="/data/pseudos/"$DATASET"/"$EXP"/"$NAME-$LC
HC_Labels="/data/pseudos/"$DATASET"/"$EXP"/"$NAME-$HC
FW_Labels="/data/pseudos/"$DATASET"/"$EXP"/"$NAME-$HC-FW
BW_Labels="/data/pseudos/"$DATASET"/"$EXP"/"$NAME-$HC-BW
Merged="/data/pseudos/"$DATASET"/"$EXP"/"$NAME-$HC-Merged
pseudo_labels="/data/pseudos/"$DATASET"/"$EXP"/"$NAME-$HC-pseudos-without-conf

Fail_pass_directory="/data/pseudos/"$DATASET"/"$EXP"/"$NAME"_fail_pass_lists"
Failed=$Fail_pass_directory"/failed_images_list.txt"
Passed=$Fail_pass_directory"/passed_images_list.txt"

Name_YO=$NAME-$HC-YOLO
Name_FW=$NAME-$HC-Forward
Name_BW=$NAME-$HC-Backward
Name_RE=$NAME-$HC-Recovered

# Configurations
print_info=false
generate_pseudos_low=false
generate_pseudos_high=false
move_low_confidence_to_pseudos=false
move_high_confidence_to_pseudos=false
copy_images_to_LC_for_deepSORT=false
recover_labels_forward_pass=true
recover_labels_backward_pass=true
merge_forward_backward=true
filter_images=true
prepare_dataset=true
copy_images_for_retraining=true

echo ' '
echo 'Current Working Directory '$(pwd)
echo ' '
 
[ ! -d "/data/pseudos/"$DATASET ] && mkdir "/data/pseudos/"$DATASET && echo "Making directory /data/pseudos/"$DATASET
[ ! -d "/data/pseudos/"$DATASET"/"$EXP ] && mkdir "/data/pseudos/"$DATASET"/"$EXP && echo "Making directory/data/pseudos/"$DATASET"/"$EXP
[ ! -d $Fail_pass_directory ] && mkdir $Fail_pass_directory && echo "Making directory "$Fail_pass_directory

if $print_info
then
	echo ' '
	echo "Name     : "$NAME
	echo ' '
	echo ' '
	echo "Data     : "$DATA
	echo "Data G T : "$DATA_GT
	echo "Data List: "$DATA_list
	echo ' '
	echo ' '
	echo "YOLO     : "$YOLO
	echo "Weights  : "$Yolo_weights
	echo "DeepSORT : "$DS
	echo ' '
	echo ' '
	echo "Lower  Confidence Threshold : "$LC
	echo "Higher Confidence Threshold : "$HC
	echo "IOU    Confidence Threshold : "$IoU
	echo ' '
	echo ' '
	echo "Low Confidence Labels   :" $LC_Labels
	echo "High Confidence Labels  :"$HC_Labels
	echo "Recovered Labels Fwd    :" $FW_Labels
	echo "Recovered Labels Bkd    :" $BW_Labels
	echo "Merged Labels Fwd+Bkd   :" $Merged
	echo "Failed Images List      :" $Failed
	echo "Passed Images List      :" $Passed
else
	echo ' '
	echo "Skip printing info."
	echo ' '
fi

echo ' '
echo ' '
echo ' '
echo ' '

if $generate_pseudos_low
then
	if [ -d $YOLO/runs/val/$NAME-$LC ]
	then
		echo ' '
		echo ' '
		echo "Make Backup of old runs/val directory"
		[ -d $YOLO/runs/val/$NAME-$LC-bkp ] && rm -rf $YOLO/runs/val/$NAME-$LC-bkp
		echo mv $YOLO/runs/val/$NAME-$LC $YOLO/runs/val/$NAME-$LC-bkp
		mv $YOLO/runs/val/$NAME-$LC $YOLO/runs/val/$NAME-$LC-bkp
	else
		echo ' '
		echo ' '
		echo "No old directory at "$YOLO/runs/val/$NAME-$LC". Skip creating backup"
	fi
	echo ' '
	echo ' '
	echo ' '
	echo "Run YOLO for Low Confidence Labels"
	echo python $YOLO/val.py 
	echo        --weights $Yolo_weights --device $Device --name $NAME-$LC --save-txt --data $DATA  --conf-thres $LC --iou-thres $IoU --save-conf --clean_labels
	echo ' '
	python $YOLO/val.py --weights $Yolo_weights --device $Device --name $NAME-$LC --save-txt --data $DATA  --conf-thres $LC --iou-thres $IoU --save-conf --clean_labels
else
	echo ' '
	echo "Skip generating Low Confidence Labels"
	echo ' '
fi
echo " "
if $generate_pseudos_high
then
	if [ -d $YOLO/runs/val/$NAME-$HC ]
	then
		echo ' '
		echo ' '
		echo "Make Backup of old runs val directory"
		[ -d $YOLO/runs/val/$NAME-$HC-bkp ] && rm -rf $YOLO/runs/val/$NAME-$HC-bkp
		echo mv $YOLO/runs/val/$NAME-$HC $YOLO/runs/val/$NAME-$HC-bkp
		mv $YOLO/runs/val/$NAME-$HC $YOLO/runs/val/$NAME-$HC-bkp
	else
		echo ' '
		echo ' '
		echo "No old directory at "$YOLO/runs/val/$NAME-$HC". Skip creating backup"
	fi
	echo ' '
	echo ' '
	echo ' '
	echo "Run YOLO for High Confidence Labels"
	echo python $YOLO/val.py 
	echo --weights $Yolo_weights --device $Device --name $NAME-$HC --save-txt --data $DATA  --conf-thres $HC --iou-thres $IoU --save-conf --clean_labels
	echo " "
	python $YOLO/val.py --weights $Yolo_weights --device $Device --name $NAME-$HC --save-txt --data $DATA  --conf-thres $HC --iou-thres $IoU --save-conf --clean_labels
else
	echo " "
	echo "Skip generating High Confidence Labels"
	echo ' '
fi

echo " "
# rm -rf /data/pseudos/$NAME*
if $move_low_confidence_to_pseudos
then
	if [ -d $LC_Labels ]
	then
		echo ' '
		echo ' '
		echo "Make Backup of old labels - Low Confidence"
		[ -d $LC_Labels-bkp ] && rm -rf $LC_Labels-bkp
		echo "mv "$LC_Labels $LC_Labels-bkp
		mv $LC_Labels $LC_Labels-bkp
	fi

	echo ' '
	echo ' '
	echo "Move new Low confidence labels to the designated folder"
	echo mv $YOLO/runs/val/$NAME-$LC/labels $LC_Labels
	mv $YOLO/runs/val/$NAME-$LC/labels $LC_Labels
else
	echo ' '
	echo "Skip moving low confidence labels from YOLO directory to pseudos directory"
	echo ' '
fi



if $move_high_confidence_to_pseudos
then
	if [ -d $HC_Labels ]
	then
		echo ' '
		echo ' '
		echo "Make Backup of old labels - High Confidence"
		[ -d $HC_Labels-bkp ] && rm -rf $HC_Labels-bkp
		echo mv $HC_Labels $HC_Labels-bkp
		mv $HC_Labels $HC_Labels-backup
	fi
	echo ' '
	echo ' '
	echo "Move new High confidence labels to the designated folder"
	echo mv $YOLO/runs/val/$NAME-$HC/labels $HC_Labels
	mv $YOLO/runs/val/$NAME-$HC/labels $HC_Labels
else
	echo ' '
	echo "Skip moving high confidence labels from YOLO directory to pseudos directory"
	echo ' '
fi


if $copy_images_to_LC_for_deepSORT
then
	echo ' '
	echo ' '
	echo "Copying images in the pesudo labels for running DeepSORT"
	echo xargs -a $DATA_list cp -t $LC_Labels
	xargs -a $DATA_list cp -t $LC_Labels
else
	echo ' '
	echo "Skip copying images to Low Confidence labels to run DeepSORT - Recovery Algorithm."
	echo ' '
fi


if $recover_labels_forward_pass
then
	if [ -d $FW_Labels ]
	then
		echo ' '
		echo ' '
		echo 'Make backup of old Forward labels if any'
		[ -d $FW_Labels-bkp ] && rm -rf $FW_Labels-bkp
		echo mv $FW_Labels  $FW_Labels-bkp
		mv $FW_Labels  $FW_Labels-bkp
	fi
	echo ' '
	echo ' '
	echo 'Recover labels - Forward'
	echo python trackv2_from_file.py 
	echo        --source $LC_Labels --device $Device --output $FW_Labels --conf-thres $HC --iou-thres $IoU --save-txt --exist-ok --deep_sort_model $DS --yolo_model $Yolo_weights
	echo ' '
	python trackv2_from_file.py --source $LC_Labels --device $Device --output $FW_Labels --conf-thres $HC --iou-thres $IoU --save-txt --exist-ok --deep_sort_model $DS --yolo_model $Yolo_weights
else
	echo ' '
	echo "Skip recovering labels in Forward pass."
	echo ' '
fi

if $recover_labels_backward_pass
then
	if [ -d $BW_Labels ]
	then
		echo ' '
		echo ' '
		echo 'Make backup of old Reverse labels if any'
		[ -d $BW_Labels-bkp ] && rm -rf $BW_Labels-bkp
		echo mv $BW_Labels  $BW_Labels-bkp
		mv $BW_Labels  $BW_Labels-bkp
	fi

	echo ' '
	echo ' '
	echo 'Recover labels - Backward'
	echo python trackv2_from_file.py 
	echo          --source $LC_Labels --device $Device --output $BW_Labels --conf-thres $HC --iou-thres $IoU --save-txt --exist-ok --deep_sort_model $DS --yolo_model $Yolo_weights --reverse
	echo ' '
	python trackv2_from_file.py --source $LC_Labels --device $Device --output $BW_Labels --conf-thres $HC --iou-thres $IoU --save-txt --exist-ok --deep_sort_model $DS --yolo_model $Yolo_weights --reverse
else
	echo ' '
	echo "Skip recovering labels in Backward pass."
	echo ' '
fi

if $merge_forward_backward
then
	if [ -d $Merged ]
	then
		echo ' '
		echo ' '
		echo 'Make backup of old Merged labels if any'
		[ -d $Merged-bkp ] && rm -rf $Merged-bkp
		echo mv $Merged  $Merged-bkp
		mv $Merged  $Merged-bkp
	fi
	echo ' '
	echo ' '
	echo 'Merge Forward and Backward'
	echo python merge_forward_backward_v2.py 
	echo          --forward $FW_Labels --backward $BW_Labels --merged $Merged
	echo ' '
	python merge_forward_backward_v2.py --forward $FW_Labels --backward $BW_Labels --merged $Merged
else
	echo ' '
	echo "Skip merging Forward and Backward labels."
	echo ' '
fi



if $filter_images
then
	if [ -d $Failed ] || [ -d $Passed ]
	then
		echo ' '
		echo ' '
		echo 'Make backup of old filtered images labels if any'
		[ -d $Failed-bkp ] && rm -rf $Failed-bkp
		[ -d $Passed-bkp ] && rm -rf $Passed-bkp
		echo mv $Failed  $Failed-bkp
		echo mv $Passed  $Passed-bkp
		mv $Failed  $Failed-bkp
		mv $Passed  $Passed-bkp
	fi
	[ -d $Fail_pass_directory-bkp ] && rm -rf $Fail_pass_directory-bkp
	[ -d $Fail_pass_directory ] && mv $Fail_pass_directory $Fail_pass_directory-bkp
	[ ! -d $Fail_pass_directory ] && mkdir $Fail_pass_directory
	echo ' '
	echo ' '
	echo 'Filter images'
	echo python filter_images_into_passed_failed.py 
	echo          --data $DATA --name $NAME-Recovery --gt $DATA_GT --saved_predictions $HC_Labels --failed_imgs_list $Failed --good_imgs_list $Passed --exist-ok
	echo ' '
	python filter_images_into_passed_failed.py --data $DATA --name $NAME-Recovery --gt $DATA_GT --saved_predictions $HC_Labels --failed_imgs_list $Failed --good_imgs_list $Passed --exist-ok
else
	echo ' '
	echo "Skip filtering labels into good and bad images."
	echo ' '
fi


if $prepare_dataset
then
	echo ' '
	echo ' '
	echo 'Prepare data for retraining images'
	echo python prepare_filtered_images.py
	echo          --GT-path $DATA_GT --PD-path $Merged --failed_GT $Failed --passed_GT $Passed
	echo ' '
	python prepare_filtered_images.py --GT-path $DATA_GT --PD-path $Merged --failed_GT $Failed --passed_GT $Passed

	echo "Files in" $Merged $(ls $Merged | wc -l)
	echo "Files in" $Merged-no-conf $(ls $Merged-no-conf | wc -l)
else
	echo ' '
	echo 'Skip preparing dataset files / label lists'
	echo ' '
fi

if $copy_images_for_retraining
then
	echo ' '
	echo ' '
	echo ' '
	echo ' '
	echo "Copying images in the pesudo labels for re-training"
	echo xargs -a $DATA_list cp -t $Merged-no-conf
	xargs -a $DATA_list cp -t $Merged-no-conf
else
	echo ' '
	echo 'Skip copying images for re-training'
	echo ' '
fi
