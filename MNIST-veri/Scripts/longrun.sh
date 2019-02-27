for unused in {100..200}
do
	./fulltest.sh $unused $unused > garbage.txt &
done
