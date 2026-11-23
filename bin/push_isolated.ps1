<#
.SYNOPSIS
    Push the isolated FashionFit-PK FYP repository to GitHub (fyp26fashionfitpk)
    WITHOUT touching or conflicting with global Git credentials.
.DESCRIPTION
    GitHub disabled account-password pushes in 2021. Provide a Personal Access
    Token (PAT) when prompted, or configure an isolated SSH key via core.sshCommand.
    Credentials are never written to git config or disk.
#>
param(
    [string]$RepoName = "fashionfit-pk",
    [string]$GitUser = "fyp26fashionfitpk"
)

$ErrorActionPreference = "Stop"
$secToken = Read-Host -Prompt "Enter GitHub Personal Access Token (PAT) for $GitUser (input hidden)" -AsSecureString
$bstr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secToken)
$token = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)

try {
    # 1. Create repository if it does not already exist
    $headers = @{
        Authorization = "token $token"
        Accept        = "application/vnd.github+json"
        "User-Agent"  = "fyp26-isolated-pusher"
    }
    $body = @{
        name        = $RepoName
        description = "FashionFit-PK: Culturally-Aware AI Virtual Try-On for Pakistani Ethnic Wear (ITU Fall 2026 FYP)"
        private     = $false
    } | ConvertTo-Json

    try {
        Invoke-RestMethod -Uri "https://api.github.com/user/repos" -Method Post -Headers $headers -Body $body -ContentType "application/json" | Out-Null
        Write-Host "[OK] Created GitHub repository https://github.com/$GitUser/$RepoName"
    } catch {
        Write-Host "[INFO] Repository may already exist or API returned: $($_.Exception.Message)"
    }

    # 2. Push branch + tags using ephemeral header (NEVER stored in .git/config or global credential manager)
    $authB64 = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("${GitUser}:${token}"))
    $cleanUrl = "https://github.com/$GitUser/$RepoName.git"

    git config --local credential.helper ""
    if (git remote | Select-String -Quiet "^origin$") {
        git remote set-url origin $cleanUrl
    } else {
        git remote add origin $cleanUrl
    }

    git -c "http.extraHeader=AUTHORIZATION: basic $authB64" push -u origin main
    git -c "http.extraHeader=AUTHORIZATION: basic $authB64" push origin --tags

    Write-Host "[SUCCESS] Pushed main branch and tags to $cleanUrl with ZERO global git credential changes."
} finally {
    $token = $null
    [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
}
