$data = "cora"
$rounds = 2

$node_selection_strategy = "directly_affected_nodes"

$Jobs = @()

$kList = 20# #300, 250, 200, 150, 100, 50, 25, 20, 15, 10, 5

Foreach ($k in $kList) {
    For ($i = 0; $i -le $rounds; $i++) {
        $Jobs += Start-Process python "train.py --debug true --dataset $data --nodeaug true --enable_K_NodeAug true --node_selection_strategy $node_selection_strategy --k $k --file_path ./gcn_k_testing/$data`_$node_selection_strategy`_$k`_$i.txt" -NoNewWindow -PassThru
        Write-Output "Started job for $data - $node_selection_strategy - $i"
    }
    $Jobs | Wait-Process
    Write-Output "Done jobs for batch $k"
}

Write-Output "Done all jobs"