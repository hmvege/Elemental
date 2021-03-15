# Create single video, Sound of the ELEMENTS, containing all of the elements.

clip_length=10
read -p "Set number of threads: " threads
read -p "Output folder: " output_folder
if [ ! -d $output_folder ]; then
    echo "Directory does not exist: "$output_folder
    exit
fi

mkdir -v -p $output_folder

emissison_spectra_folder=$output_folder"/spectra_images_watermarked"
mkdir -v -p $emissison_spectra_folder

audio_clips_folder=$output_folder"/audio_clips"
mkdir -v -p $audio_clips_folder

audio_clips_folder_resampled=$output_folder"/audio_clips_resampled"
mkdir -v -p $audio_clips_folder_resampled

audio_sequence_output=$output_folder"/audio_output"
mkdir -v -p $audio_sequence_output

# Using a short script for generating new, short element sounds.
echo "Generating short "$clip_length"s audio clips for each element"
for element in $(
    python scripts/get_viable_elements.py get-elements --return-type short
)
do
    echo "Generating $element"
    elemental \
        $element \
        -lf "spectras/"$element".dat" \
        -ln $clip_length \
        -p -np $threads \
        -ev 0.1 \
        -sr 48000 \
        -o $audio_clips_folder
done

#Generates emission spectra for each element.
echo "Creating emission spectra in: $emissison_spectra_folder"
python scripts/generate_emission_spectra.py \
    --output_folder $emissison_spectra_folder \
    --element_watermark \
    --spectra_option "empirical"
echo "Emission spectra generation complete."

# Remove the element names from the spectra
echo "Stripping element name from files in $emissison_spectra_folder"
for f in $(ls $emissison_spectra_folder)
do
    mv \
        $emissison_spectra_folder/$f \
        $emissison_spectra_folder/"$(echo $f | sed "s/[A-Za-z]*_//")"
done

# Re-samples audio from 64 to 16 bits.
echo "Resampling spectra sounds to: " $audio_clips_folder_resampled
python scripts/get_viable_elements.py get-elements --return-type short \
    | xargs -I % -n 1 -P $threads \
    ffmpeg \
        -i $audio_clips_folder"/"%"_"$clip_length"sec.wav" \
        -y \
        -hide_banner \
        -loglevel error \
        -sample_fmt s16 \
        -ar 44100 \
        $audio_clips_folder_resampled"/"%"_"$clip_length"sec_16b.wav"
echo "Re-sampled spectra from 64 to 16 bit."

# Joins together the short element clips in 8 second segments.
echo "Creating joined audio spectra in: $audio_sequence_output"
python3 \
    scripts/generate_elements_audio.py \
    $audio_clips_folder_resampled \
    $audio_sequence_output \
    $clip_length

# Re-samples audio from wav to mp3.
echo "Converting from $input_wav to $output_mp3"
ffmpeg \
    -y \
    -hide_banner \
    -loglevel warning \
    -i \
    $audio_sequence_output/"sound_of_the_elements.wav" \
    -b:a 320k \
    -f mp3 \
    $audio_sequence_output/"sound_of_the_elements.mp3"

# Generates the full movie of all the mp3 files.
echo "Retrieving spectra images from: $emissison_spectra_folder"
ffmpeg \
    -y \
    -hide_banner \
    -framerate 1/8 \
    -i $emissison_spectra_folder/"%03d.png" \
    -i $audio_sequence_output/"sound_of_the_ELEMENTS.mp3" \
    -vcodec libx264 \
    -r 1 \
    -acodec copy \
    $output_folder/sound_of_the_ELEMENTS.mp4
