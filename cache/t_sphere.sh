wget -r -np -nd -A txt http://neilsloane.com/sphdesigns/dim3/des.3.240.21.txt

for file in des*.txt; do
    i=${file%*.txt}
    i=${i:6}
    NODES=$(printf "%05d" ${i%.*})
    ORDER=$(printf "hs%03d" ${i#*.})
    mv ${file} $(printf $ORDER.$NODES)
done