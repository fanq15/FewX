rm support_feature.pkl
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 fsod_train_net.py --num-gpus 4 \
	--config-file configs/fsod/R_50_C4_1x.yaml 2>&1 | tee log/fsod_train_log.txt

#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 tools/train_net.py --num-gpus 4 \
#	--config-file configs/fsod/R_50_C4_1x.yaml \
#	--eval-only MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_final.pth 2>&1 | tee log/fsod_test_log.txt

rm support_feature.pkl
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 fsod_train_net.py --num-gpus 4 \
	--config-file configs/fsod/finetune_R_50_C4_1x.yaml 2>&1 | tee log/fsod_finetune_train_log.txt
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 fsod_train_net.py --num-gpus 4 \
	--config-file configs/fsod/finetune_R_50_C4_1x.yaml \
	--eval-only MODEL.WEIGHTS ./output/fsod/finetune_dir/R_50_C4_1x/model_final.pth 2>&1 | tee log/fsod_finetune_test_log.txt

#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 fsod_train_net.py --num-gpus 4 \
#	--config-file configs/fsod/finetune_R_50_C4_1x.yaml \
#	--eval-only MODEL.WEIGHTS ./output/fsod/finetune_dir/R_50_C4_1x/model_final.pth 2>&1 | tee log/fsod_finetune_test_log.txt

