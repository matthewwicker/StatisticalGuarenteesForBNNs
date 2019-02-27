for i in VI/ MCMC/ MCDropout/
do
for (( IMNUM=$1; IMNUM<=$2; IMNUM++ ))
do
	for EPS in 0 1 2 3 4
	do
		for M in 0 1 2 3
		do
python -W ignore RobustnessTest.py --imnum $IMNUM --sev $EPS --mode $M --path $i
		done
	done
done
done
