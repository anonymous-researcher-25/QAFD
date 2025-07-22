
source .env
db_root_directory=$DB_ROOT_DIRECTORY 

verbose=true
signature_size=100
n_gram=3
threshold=0.01

# db_id="concert_singer" # Options: all or a specific db_id

for dir in /data/dev_databases/*/
do
    dir=${dir%*/} 
    db_id=${dir##*/}
    
    python3 -u ./src/preprocess.py --db_root_directory "${db_root_directory}" \
                            --signature_size "${signature_size}" \
                            --n_gram "${n_gram}" \
                            --threshold "${threshold}" \
                            --db_id "${db_id}" \
                            --verbose "${verbose}"
    fi
done

