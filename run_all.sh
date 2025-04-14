values=(examples/*.tsp)
num=$1
make
for i in "${values[@]}"
do
    START=$(date +%s.%N)
    ./acotsp $i out.dat WORKER $num 1 2 0.5 2137
    END=$(date +%s.%N)
    ELAPSED=$(echo "$END - $START" | bc)
    echo "(WORKER) Lowest distance for: $i -> $(head -n 1 out.dat | cut -d ' ' -f 1) in $ELAPSED seconds"
    
    START=$(date +%s.%N)
    ./acotsp $i out.dat QUEEN $num 1 2 0.5 2137
    END=$(date +%s.%N)
    ELAPSED=$(echo "$END - $START" | bc)
    echo "(QUEEN) Lowest distance for: $i -> $(head -n 1 out.dat | cut -d ' ' -f 1) in $ELAPSED seconds"
done
rm out.dat
