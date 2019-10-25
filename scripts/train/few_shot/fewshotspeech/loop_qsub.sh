for shot in 1 5 10
do

	i=1
	
	for way in  2 2 3 4

	do 
        	qsub -v shot=$shot,way=$way,flag=$i qsub_train.sh
    		i=$((i+1))

	done


done
