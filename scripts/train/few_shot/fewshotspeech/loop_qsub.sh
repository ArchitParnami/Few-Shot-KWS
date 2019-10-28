id=1
for way in 2
do
	for shot in  1 5 10
	do 
        for i in 0 1 2 3 4 5 6 7
        do
            qsub -v way=$way,shot=$shot,flag=$i,id=$id qsub_train.sh
            id=$((id+1))
        done
	done


done
