
# Enter json file id as argument e.g. 1 or 2 or ...
echo "Swapping model results $1"
local_file="model_results.json"
data_dir="../../../../data3/xray_data_augmentation/test_data/"
echo "$local_file"
cp "$data_dir""model_results""$1"".json" "$local_file"
echo "model_results""$1"".json" > "last_result_loaded.txt"