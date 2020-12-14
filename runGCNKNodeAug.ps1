$dataset = "cora", "citeseer", "pubmed"
$rounds = 10

$node_selection_strategies = "random", "directly_affected_nodes", "one_hop_neighbors"

$Jobs = @()

Foreach ($data in $dataset) {
    For ($i = 0; $i -le $rounds; $i++) {
        Foreach ($node_selection_strategy in $node_selection_strategies) {
            $Jobs += Start-Process python "train.py --debug false --dataset $data --nodeaug true --enable_K_NodeAug true --node_selection_strategy $node_selection_strategy --file_path ./gcn_k_node_aug_data/$data`_$node_selection_strategy`_$i.txt" -NoNewWindow -PassThru
            Write-Output "Started job for $data - $node_selection_strategy - $i"
        }
        $Jobs | Wait-Process
    }
    Write-Output "Done jobs for batch $i"
}

Write-Output "Done all jobs"