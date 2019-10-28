for way in 2
do	
	for shot in  1 5 10
	do 

		for i in 0 1 2 3 4 5 6 7
		do 
			args="../run_train.py \
			--data.dataset=googlespeech --data.way=$way --data.shot=$shot \
			--data.query=5 --data.test_way=$way --data.test_shot=$shot --data.test_query=15 \
			--data.train_episodes=100 --data.test_episodes=100 \
			--model.model_name=protonet_conv --model.x_dim=1,51,40 --model.hid_dim=64 --model.z_dim=64 \
			--train.epochs=200 --train.optim_method=Adam --train.learning_rate=0.001 --train.decay_every=20 \
			--train.weight_decay=0.0 --train.patience=200  \
			--log.exp_dir=. \
			--data.cuda"

			if [ "$i" -eq "1" ]; then
				args+=" --speech.include_background"

			elif [ "$i" -eq "2" ]; then
				args+=" --speech.include_silence"

			elif [ "$i" -eq "3" ]; then
				args+=" --speech.include_unknown"

			elif [ "$i" -eq "4" ]; then
				args+=" --speech.include_background"
				args+=" --speech.include_silence"

			elif [ "$i" -eq "5" ]; then
				args+=" --speech.include_background"
				args+=" --speech.include_unknown"

			elif [ "$i" -eq "6" ]; then
				args+=" --speech.include_unknown"
				args+=" --speech.include_silence"
	
			elif [ "$i" -eq "7" ]; then
				args+=" --speech.include_background"
				args+=" --speech.include_unknown"
				args+=" --speech.include_silence"
			fi
			
			python $args

		done


	done


done


