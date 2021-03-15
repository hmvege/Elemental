# Script for creating the sound, background image and video for all of the elements.

read -p "Set number of threads: " threads

# 10 minute long video
length=600

sounds_wav="sounds"
mkdir -v $sounds_wav

converted_sounds_folder="converted_sounds_folder"
mkdir -v $converted_sounds_folder

output_movie_folder="output_mp4"
mkdir -v $output_movie_folder

# Generates the full 10 minute element clips.
for element in $(
   python scripts/get_viable_elements.py get-elements --return-type short
)
do
   echo "Creating sound for $element"
   elemental $element -lf spectras/$element.dat -ln $length -p -np $threads -sr 48000
done

# Converts wav to mp3.
python scripts/get_viable_elements.py get-elements --return-type short \
   | xargs -I % -n 1 -P $threads \
   ffmpeg \
       -y \
       -hide_banner \
       -loglevel error \
       -i $sounds_wav"/"%"_"$length"sec.wav" \
       -b:a 320k \
       -f mp3 \
       "$converted_sounds_folder/"%"_"$length"sec.mp3"

# Runs the sound generation in parallel
python scripts/get_viable_elements.py get-elements --return-type short \
    | xargs -I @@ bash -c ' \
        elem_name=$(python scripts/get_viable_elements.py elem2full @@);
        elem_ids=$(python scripts/get_viable_elements.py elem2ids @@);
        printf -v ids "%03d" $elem_ids;
        echo "Generating video for $elem_name";
        ffmpeg \
            -y \
            -hide_banner \
            -loglevel error \
            -nostdin \
            -loop 1 \
            -i "generated_emission_spectras/"$elem_name"_"$ids".png" \
            -i $0"/"@@"_"$1"sec.mp3" \
            -vcodec libx264 \
            -acodec copy \
            -shortest \
            $2"/"@@"_"$1"sec.mp4";' \
            $converted_sounds_folder \
            $length \
            $output_movie_folder

python scripts/rename_title.py
