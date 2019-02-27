for i in VI/
do
for DELT in 0
do
	for EPS in 1 2 3 4 5
	do
		for M in 5
		do
python -W ignore RobustnessTest.py --imnum $1 --eps $EPS --mode $M --path $i --delt $DELT
		done
	done
done
done
