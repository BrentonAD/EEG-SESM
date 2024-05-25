for i in {1..52}; do 
    wget -c ftp://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100295/mat_data/s$(printf "%02d" $i).mat -P "E:\s222165064\motor_imagery\raw"
done