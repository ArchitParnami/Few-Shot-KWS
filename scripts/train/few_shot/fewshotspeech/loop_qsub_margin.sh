id=1
for way in 2 3
do
	for shot in 5 
	do 
        	for alpha in 0.01 0.1 0.2 0.5
            do
                for margin in 0.1 0.2 0.5 1.0
                do
                    for i in 0
        	        do
            		    qsub -v way=$way,shot=$shot,flag=$i,id=$id,alpha=$alpha,margin=$margin qsub_train.sh
            		    id=$((id+1))
        	        done

                done

            done
            
            
	done


done
