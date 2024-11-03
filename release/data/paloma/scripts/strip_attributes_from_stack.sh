# used to remove unneeded info from dolma 100 programming languages
stack_dir=$1
output_dir=$2

for split in val test; do
    mkdir -p $output_dir/$split
    for file in $(ls $stack_dir/$split); do
        echo "Stripping attributes from $file"
        zcat < $stack_dir/$split/$file | jq -c 'del(.attributes)' | gzip > $output_dir/$split/$file
    done
done
