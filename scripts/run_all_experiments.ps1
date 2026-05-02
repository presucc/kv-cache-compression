param(
    [string]$Model = "EleutherAI/pythia-70m",
    [string]$Pg19Text = "data/pg19_raw/test/10146.txt",
    [string]$WikitextText = "data/wikitext_validation.txt",
    [int]$MaxTokens = 1024,
    [int]$WindowSize = 256,
    [int]$SinkSize = 4,
    [int]$ImportantSize = 32,
    [string]$Device = "auto",
    [string]$DType = "auto",
    [switch]$SkipLatency
)

$ErrorActionPreference = "Stop"

if (-not $env:PYTHONPATH) {
    $env:PYTHONPATH = "src"
} elseif ($env:PYTHONPATH -notlike "*src*") {
    $env:PYTHONPATH = "src;$env:PYTHONPATH"
}

New-Item -ItemType Directory -Force -Path "results" | Out-Null
$Methods = @("dense", "sliding_window", "streamingllm", "lm_infinite", "h2o", "scissorhands", "tova", "snapkv", "pyramidkv", "asw_kv")

Write-Host "== PG-19 PPL =="
python scripts/run_ppl.py `
    --model $Model `
    --text-file $Pg19Text `
    --max-tokens $MaxTokens `
    --methods $Methods `
    --window-size $WindowSize `
    --sink-size $SinkSize `
    --important-size $ImportantSize `
    --device $Device `
    --dtype $DType `
    --output results/ppl_pg19_1024_all_kv.json

Write-Host "== Wikitext-2 PPL =="
python scripts/run_ppl.py `
    --model $Model `
    --text-file $WikitextText `
    --max-tokens $MaxTokens `
    --methods $Methods `
    --window-size $WindowSize `
    --sink-size $SinkSize `
    --important-size $ImportantSize `
    --device $Device `
    --dtype $DType `
    --output results/ppl_wikitext_1024_all_kv.json

Write-Host "== PG-19 important_size ablation =="
$ablation = @()
foreach ($k in 0, 16, 32, 64) {
    $out = "results/ablation_pg19_asw_important_$k.json"
    python scripts/run_ppl.py `
        --model $Model `
        --text-file $Pg19Text `
        --max-tokens $MaxTokens `
        --methods asw_kv `
        --window-size $WindowSize `
        --sink-size $SinkSize `
        --important-size $k `
        --device $Device `
        --dtype $DType `
        --output $out
    $row = Get-Content -Raw -LiteralPath $out | ConvertFrom-Json
    $row | Add-Member -NotePropertyName important_size -NotePropertyValue $k -Force
    $ablation += $row
}
$ablation | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 -LiteralPath "results/ablation_pg19_important_size.json"

if (-not $SkipLatency) {
    Write-Host "== GPU latency =="
    python scripts/run_latency.py `
        --model $Model `
        --text-file $Pg19Text `
        --max-prompt-tokens 512 `
        --max-new-tokens 64 `
        --methods $Methods `
        --window-size $WindowSize `
        --sink-size $SinkSize `
        --important-size $ImportantSize `
        --device cuda `
        --dtype float16 `
        --output results/latency_pg19_gpu_512_64_all_kv.json
}

Write-Host "== Summary =="
python scripts/summarize_results.py results/ppl_pg19_1024_all_kv.json --columns method ppl max_retained_tokens average_retained_tokens
python scripts/summarize_results.py results/ppl_wikitext_1024_all_kv.json --columns method ppl max_retained_tokens average_retained_tokens
python scripts/summarize_results.py results/ablation_pg19_important_size.json --columns important_size ppl max_retained_tokens average_retained_tokens
