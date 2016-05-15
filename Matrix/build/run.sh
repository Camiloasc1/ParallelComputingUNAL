#!/bin/sh

TIMEFORMAT=%R
I=0
echo -n > Time.dat
echo -n > Speedup.dat
for N in 2 4 8 16 32 64 128 256 512 1024
do
    echo "N = $N"
    echo "# N = $N (index $I)" >> Time.dat
    echo "# N = $N (index $I)" >> Speedup.dat
    I=$((I + 1))
    T1=0
    for P in 1 2 4 8 16 32 64 128 256 512 1024
    do
        TIME=$( { time ./$1 $P $N > /dev/null; } 2>&1 )
        if [ "$P" = "1" ]; then
            T1=$TIME
        fi
        echo $P $TIME  >> Time.dat
        echo $P $(echo $T1 / $TIME | bc -l)  >> Speedup.dat
    done
    echo  >> Time.dat
    echo  >> Time.dat
    echo  >> Speedup.dat
    echo  >> Speedup.dat
done

gnuplot -p -e "load 'PlotTime.p'"
gnuplot -p -e "load 'PlotSpeedup.p'"
