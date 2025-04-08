values=(examples/*.tsp)
for i in "${values[@]}"
do
    ./acotsp $i out.dat WORKER 1000 1 2 0.5 2137
    echo "(WORKER) Lowest distance for: $i -> $(head -n 1 out.dat | cut -d ' ' -f 1)"
    ./acotsp $i out.dat QUEEN 1000 1 2 0.5 2137
    echo "(QUEEN) Lowest distance for: $i -> $(head -n 1 out.dat | cut -d ' ' -f 1)"
done
rm out.dat
