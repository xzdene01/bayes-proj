#!/usr/bin/env bash
set -e

# Options: VCTK, LIBRISPEECH, CNCELEB1
# DATASET="VCTK"
# DATASET="LIBRISPEECH"
DATASET="CNCELEB1"

if [ "$DATASET" = "VCTK" ]; then
    OUT_DIR="VCTK"
    URL="https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
    ARCHIVE_NAME="VCTK-Corpus-0.92.zip"
    EXTRACT_CMD="unzip -q $ARCHIVE_NAME -d $OUT_DIR"
elif [ "$DATASET" = "LIBRISPEECH" ]; then
    OUT_DIR="LibriSpeech"
    URL="https://www.openslr.org/resources/12/train-other-500.tar.gz"
    ARCHIVE_NAME="train-other-500.tar.gz"
    EXTRACT_CMD="tar -xzf $ARCHIVE_NAME -C $OUT_DIR --strip-components=2"
elif [ "$DATASET" = "CNCELEB1" ]; then
    OUT_DIR="CN-Celeb1"
    URL="https://openslr.elda.org/resources/82/cn-celeb_v2.tar.gz"
    ARCHIVE_NAME="cn-celeb_v2.tar.gz"
    EXTRACT_CMD="tar -xzf $ARCHIVE_NAME -C $OUT_DIR --strip-components=1"
else
    echo "Error: Unknown dataset '$DATASET'. Use 'VCTK' or 'LIBRISPEECH'."
    exit 1
fi

if [ -d "$OUT_DIR" ]; then
    echo "Error: Directory '$OUT_DIR' already exists. Remove it or choose another name."
    exit 1
fi

if [ -f "$ARCHIVE_NAME" ]; then
    echo "Archive already downloaded: $ARCHIVE_NAME"
else
  echo "Downloading $DATASET dataset ..."
  wget -O "$ARCHIVE_NAME" "$URL"
fi

echo "Extracting ..."
mkdir -p $OUT_DIR
$EXTRACT_CMD

# Postprocess the datasets if needed
if [ "$DATASET" = "VCTK" ]; then
  OLD_DIR="${OUT_DIR}_old"
  mv $OUT_DIR $OLD_DIR
  mkdir -p $OUT_DIR

  mv $OLD_DIR/wav48_silence_trimmed/* $OUT_DIR

  rm -rf $OLD_DIR
elif [ "$DATASET" = "CNCELEB1" ]; then
  UNFILTERED_DIR="${OUT_DIR}_unfiltered"
  mv $OUT_DIR $UNFILTERED_DIR
  mkdir -p "$OUT_DIR"

  # Filter CN-Celeb1 speakers with under 10 utts
  echo "Filtering speakers with <10 recordings ..."
  find "$UNFILTERED_DIR/data" -mindepth 1 -maxdepth 1 -type d | while read spk; do
      count=$(find "$spk" -type f -name '*.flac' | wc -l)
      if [ "$count" -ge 10 ]; then
          cp -r "$spk" "$OUT_DIR"
      fi
  done

  rm -rf $UNFILTERED_DIR

  echo "Done. Filtered dataset saved to: $OUT_DIR"
else
  echo "Done. Extracted into: $OUT_DIR"
fi

