# R.H. Hardin and N.J.A. Sloane http://neilsloane.com/sphdesigns/
wget -r -np -nd -A txt http://neilsloane.com/sphdesigns/dim3/des.3.240.21.txt

for file in *.txt; do
    i=${file%*.txt}
    i=${i:6}
    NODES=$(printf "%05d" ${i%.*})
    ORDER=$(printf "hs%03d" ${i#*.})
    mv ${file} $(printf $ORDER.$NODES)
done

# Rob Womersley with symmetry https://web.maths.unsw.edu.au/~rsw/Sphere/EffSphDes/index.html
# Approx 10**3
wget -r -np -nd --no-check-certificate -O ss045.01038 -A txt https://web.maths.unsw.edu.au/~rsw/Sphere/Points/SS/SS31-Mar-2016/ss045.01038
# Approx 10**4
wget -r -np -nd --no-check-certificate -O ss141.10014 -A txt https://web.maths.unsw.edu.au/~rsw/Sphere/Points/SS/SS31-Mar-2016/ss141.10014
# Approx 5*10**4
wget -r -np -nd --no-check-certificate -O ss325.52978 -A txt https://web.maths.unsw.edu.au/~rsw/Sphere/Points/SS/SS31-Mar-2016/ss325.52978 

# Without symmetry https://web.maths.unsw.edu.au/~rsw/Sphere/EffSphDes/sf.html
# Approx 10**3
wget -r -np -nd --no-check-certificate -O sf044.01014 -A txt https://web.maths.unsw.edu.au/~rsw/Sphere/Points/SF/SF29-Nov-2012/sf044.01014
# Approx 10**4
wget -r -np -nd --no-check-certificate -O sf141.10083 -A txt https://web.maths.unsw.edu.au/~rsw/Sphere/Points/SF/SF29-Nov-2012/sf141.10083

