#!/bin/bash
run_models(){
	for d in ./models/*/ ; do
		if [ -d ${d} ]; then
			MODEL=${d%/}
			MODEL=${MODEL##*/}
			echo ${MODEL}
			CUDA_VISIBLE_DEVICES=0 python3 main.py --model_name=${MODEL} \
			--phase=inference --ii_dir ../val_large/*0.jpg \
			--save_dir ../resize_results/${MODEL}_1000_1_w_embeddings \
			--reencodes=100 --reencode_steps=1 --embeddings \
			--x_image_size=768x512
		fi
	done
}

echo "$(run_models)"

