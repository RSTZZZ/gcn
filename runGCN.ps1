
$rounds = 10

For ($i=0; $i -le $rounds; $i++) {
    python train.py | tee output_$i.txt
}