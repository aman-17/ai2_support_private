export NUMBER=$1
cat revisions-${NUMBER}.txt | while read branch; do
  echo "$branch"

  if grep -q "$branch" updated_branches-${NUMBER}.txt
    then
      echo "present"
    else
      ./upload_sf.sh $branch ~/hf/subset-${NUMBER}/OLMo-7B
      echo $branch >> updated_branches-${NUMBER}.txt
    fi
done
