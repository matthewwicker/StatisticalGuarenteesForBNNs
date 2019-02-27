for i in MCMC/ MCDropout/ VI/
do
for (( IMNUM=$1; IMNUM <= $2; IMNUM++  ))
do
	for EPS in 2 3 4
	do
	for DELT in 2 4
	do
		for M in 5
		do
python -W ignore RobustnessTest.py --imnum $IMNUM --eps $EPS --delt $DELT --mode $M --path $i
		done
	done
	done
done
done
