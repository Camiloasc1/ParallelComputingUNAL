reset
set title "Time of execution"
set ylabel "Time (s)" 
set xlabel "Processes or Threads (#)" 

set style line 1 lc rgb '#000000' lt 1 lw 2 pt 7 ps 1.5
set style line 2 lc rgb '#000088' lt 1 lw 2 pt 7 ps 1.5
set style line 3 lc rgb '#0000FF' lt 1 lw 2 pt 7 ps 1.5
set style line 4 lc rgb '#008800' lt 1 lw 2 pt 7 ps 1.5
set style line 5 lc rgb '#008888' lt 1 lw 2 pt 7 ps 1.5
set style line 6 lc rgb '#0088FF' lt 1 lw 2 pt 7 ps 1.5
set style line 7 lc rgb '#00FF00' lt 1 lw 2 pt 7 ps 1.5
set style line 8 lc rgb '#00FF88' lt 1 lw 2 pt 7 ps 1.5
set style line 9 lc rgb '#00FFFF' lt 1 lw 2 pt 7 ps 1.5
set style line 10 lc rgb '#FF0000' lt 1 lw 2 pt 7 ps 1.5

plot 'Time.dat' index 0 with linespoints ls 1 title "N = 2", \
     ''         index 1 with linespoints ls 2 title "N = 4", \
     ''         index 2 with linespoints ls 3 title "N = 8", \
     ''         index 3 with linespoints ls 4 title "N = 16", \
     ''         index 4 with linespoints ls 5 title "N = 32", \
     ''         index 5 with linespoints ls 6 title "N = 64", \
     ''         index 6 with linespoints ls 7 title "N = 128", \
     ''         index 7 with linespoints ls 8 title "N = 256", \
     ''         index 8 with linespoints ls 9 title "N = 512", \
     ''         index 9 with linespoints ls 10 title "N = 1024"
