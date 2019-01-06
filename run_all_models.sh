#!/bin/bash
run_models(){
	for d in ./models/*/ ; do
		if [ -d ${d} ]; then
			MODEL=${d%/}
			MODEL=${MODEL##*/}
			echo ${MODEL}
			CUDA_VISIBLE_DEVICES=9 python3 main.py --model_name=${MODEL} --phase=inference --image_size=2500 --ii_dir ../house2/ --save_dir ../house_results/${MODEL} --reencodes=1 --reencode_steps=1
		fi
	done
}

echo "$(run_models)"

