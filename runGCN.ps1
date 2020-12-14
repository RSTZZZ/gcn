$dataset = "cora", "citeseer", "pubmed"
$rounds = 10



For ($i=0; $i -le $rounds; $i++) {
    Foreach ($data in $dataset) {
        python train.py --debug $false --dataset $data | tee ./gcn_data/$data`_$i.txt
    }
}