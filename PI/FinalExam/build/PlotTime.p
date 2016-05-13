reset
set title "Time of execution"
set ylabel "Time (s)"
set xlabel "Threads (#)"

set logscale x 2

set style line 1 lc rgb '#000000' lt 1 lw 2 pt 7 ps 1.5

plot 'Time.dat' with linespoints ls 1 title ""
