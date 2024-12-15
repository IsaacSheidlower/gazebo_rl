# accept the start and end idxs from the command line
# and replay the episodes from start to end

# Usage: ./replay_all.sh 1 10
# This will replay episodes 1 to 10


# keep leading zeros
shopt -s extglob
start=$1
end=$2

echo "Replaying episodes from $start to $end"

for i in $(seq $start $end)
do
    # keep the leading 0
    i=$(printf "%03d" $i)
    echo "Replaying episode $i"
    python publish_saved_video.py -d user_$i --no-arm
done
