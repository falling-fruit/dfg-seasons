#!/bin/bash

FILES="ECA_blend_tg/TG_STAID*.txt"

for f in $FILES
do
	filtered=$(tail -n +22 $f | cut -d, -f1,3,4 | awk -F, '{if($2>20100101)print}' | awk -F, '{if($3>-9999)print}')	

	if [ -z "$filtered" ]
	then
		echo "skipping file"
	else
		site_id=$(echo "$filtered" | head -n 1 | cut -d, -f1)
		echo "$filtered" > "formatted_euro_weather/site_$site_id.csv"
	fi
done

