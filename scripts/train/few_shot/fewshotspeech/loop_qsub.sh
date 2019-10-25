for shot in 1 5 10
do

	i=1
	
	for way in  2 2 3 4

	do 
        ./qsub_train.sh $shot $way $i
        i=$((i+1))

	done


done