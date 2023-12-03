################# BART AMR 3.0

ROOT=data/AMR3.0/bart-large
MODEL="$ROOT"/model

python scripts/12_prepare_bart.py "$MODEL" --add-amr-tokens

python scripts/14_prepare_data.py "$ROOT"/amrtoken \
    --model "$MODEL"

python scripts/14_prepare_data.py "$ROOT"/amrtoken-dvar \
    --model "$MODEL" --detach-var

################# BART AMR 2.0

ROOT=data/AMR2.0/bart-large
MODEL="$ROOT"/model

python scripts/12_prepare_bart.py "$MODEL" --add-amr-tokens \
    --train_file data/AMR2.0/tdata_xfm/train.txt.nowiki

python scripts/14_prepare_data.py "$ROOT"/amrtoken \
    --model "$MODEL" --data-dir data/AMR2.0/tdata_xfm

python scripts/14_prepare_data.py "$ROOT"/amrtoken-dvar \
    --model "$MODEL" --detach-var --data-dir data/AMR2.0/tdata_xfm

################# BART bio amrtoken

ROOT=data/AMR3.0
MODEL="$ROOT"/bart-large/model

python scripts/14_prepare_data.py "$ROOT"/bio/amrtoken \
    --model "$MODEL" --data-dir data/AMR/bio \
    --train-file amr-release-bio-v3.0.txt \
    --dev-file amr-release-bio-v3.0.txt \
    --test-file amr-release-test-bio.txt

python scripts/14_prepare_data.py "$ROOT"/bio/amrtoken-dvar \
    --model "$MODEL" --detach-var \
    --data-dir data/AMR/bio \
    --train-file amr-release-bio-v3.0.txt \
    --dev-file amr-release-bio-v3.0.txt \
    --test-file amr-release-test-bio.txt

ROOT=data/AMR2.0
MODEL="$ROOT"/bart-large/model

python scripts/14_prepare_data.py "$ROOT"/bio/amrtoken \
    --model "$MODEL" --data-dir data/AMR/bio \
    --train-file amr-release-bio-v3.0.txt \
    --dev-file amr-release-bio-v3.0.txt \
    --test-file amr-release-test-bio.txt

python scripts/14_prepare_data.py "$ROOT"/bio/amrtoken-dvar \
    --model "$MODEL" --detach-var \
    --data-dir data/AMR/bio \
    --train-file amr-release-bio-v3.0.txt \
    --dev-file amr-release-bio-v3.0.txt \
    --test-file amr-release-test-bio.txt

################# BART new3 amrtoken

ROOT=data/AMR2.0
MODEL="$ROOT"/bart-large/model

python scripts/14_prepare_data.py "$ROOT"/new3/amrtoken \
    --model "$MODEL" --data-dir data/AMR2.0/new3 \
    --train-file amr-release-3.0-amrs-test-lorelei.txt \
    --dev-file amr-release-3.0-amrs-test-lorelei.txt \
    --test-file amr-release-3.0-amrs-test-lorelei.txt

python scripts/14_prepare_data.py "$ROOT"/new3/amrtoken-dvar \
    --model "$MODEL" --detach-var \
    --data-dir data/AMR2.0/new3 \
    --train-file amr-release-3.0-amrs-test-lorelei.txt \
    --dev-file amr-release-3.0-amrs-test-lorelei.txt \
    --test-file amr-release-3.0-amrs-test-lorelei.txt

################# BART tlp amrtoken

ROOT=data/AMR3.0
MODEL="$ROOT"/bart-large/model

python scripts/14_prepare_data.py "$ROOT"/tlp/amrtoken \
    --model "$MODEL" --data-dir data/AMR/tlp \
    --train-file amr-bank-struct-v3.0.txt \
    --dev-file amr-bank-struct-v3.0.txt \
    --test-file amr-bank-struct-v3.0.txt

python scripts/14_prepare_data.py "$ROOT"/tlp/amrtoken-dvar \
    --model "$MODEL" --detach-var \
    --data-dir data/AMR/tlp \
    --train-file amr-bank-struct-v3.0.txt \
    --dev-file amr-bank-struct-v3.0.txt \
    --test-file amr-bank-struct-v3.0.txt

