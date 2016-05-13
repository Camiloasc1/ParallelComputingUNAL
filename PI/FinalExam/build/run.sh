#!/bin/sh

TIMEFORMAT=%R
echo -n > Time.dat
echo -n > Speedup.dat
T1=0
for P in 1 2 4 8 16 32
do
TIME=$( { time ./$1 $P $2 > /dev/null; } 2>&1 )
if [ "$P" = "1" ]; then
    T1=$TIME
fi
echo $P $TIME  >> Time.dat
echo $P $(echo $T1 / $TIME | bc -l)  >> Speedup.dat
done
echo  >> Time.dat
echo  >> Speedup.dat

gnuplot -p -e "load 'PlotTime.p'"
gnuplot -p -e "load 'PlotSpeedup.p'"

