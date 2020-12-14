## Sample call
## ./runGCNNodeAug.ps1 10

$dataset = "cora", "citeseer", "pubmed"
$rounds = $args[0]

$Jobs = @()

Write-Output "Going to be running $rounds node selection strategy."

For ($i = 0; $i -le $rounds; $i++) {
    Foreach ($data in $dataset) {
        $Jobs += Start-Process python "train.py --debug true --dataset $data --nodeaug true --enable_K_NodeAug false --file_path ./gcn_node_aug_data/$data`_$i.txt" -NoNewWindow -PassThru
        Write-Output "Started job for $data - $i"
    }

    $Jobs | Wait-Process

    Write-Output "Done jobs for batch $i"
}

Write-Output "Done all jobs"

