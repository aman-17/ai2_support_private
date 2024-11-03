cat revisions.txt | while read branch; do
  echo "$branch"

  if grep -q "$branch" updated_branches.txt
    then
      echo "present"
    else
      ./upload_sf.sh $branch ~/hf/OLMo-1B
      echo $branch >> updated_branches.txt
    fi
done
