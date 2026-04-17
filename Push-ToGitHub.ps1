# GitHub Push Script for BhoomiAI
# This script automates the process of pushing BhoomiAI to GitHub

param(
    [switch]$Force = $false,
    [switch]$Verbose = $false
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Project configuration
$projectPath = "c:\Users\satye\OneDrive\Desktop\krishiai_project"
$repositoryUrl = "https://github.com/karan-nigam96/Bhoomi_AI.git"
$commitMessage = "Complete BhoomiAI upgrade: Season and Agro-Zone features integrated"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "BhoomiAI GitHub Push Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Change to project directory
Write-Host "[STEP 1] Navigating to project directory..." -ForegroundColor Yellow
try {
    Set-Location $projectPath
    Write-Host "✓ Changed to directory: $projectPath" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to change directory: $_" -ForegroundColor Red
    exit 1
}

# Step 2: Check if git repository exists
Write-Host "[STEP 2] Checking git repository..." -ForegroundColor Yellow
if (Test-Path ".git") {
    Write-Host "✓ Directory is already a git repository" -ForegroundColor Green
} else {
    Write-Host "  Initializing git repository..." -ForegroundColor Cyan
    try {
        git init
        Write-Host "✓ Git repository initialized" -ForegroundColor Green
    } catch {
        Write-Host "✗ Failed to initialize git: $_" -ForegroundColor Red
        exit 1
    }
}

# Step 3: Configure remote origin
Write-Host "[STEP 3] Configuring remote origin..." -ForegroundColor Yellow
try {
    $remoteCheck = git remote -v 2>$null | Select-String "origin"
    if ($remoteCheck) {
        Write-Host "  Updating existing remote origin..." -ForegroundColor Cyan
        git remote set-url origin $repositoryUrl
        Write-Host "✓ Remote URL updated" -ForegroundColor Green
    } else {
        Write-Host "  Adding new remote origin..." -ForegroundColor Cyan
        git remote add origin $repositoryUrl
        Write-Host "✓ Remote origin added" -ForegroundColor Green
    }
} catch {
    Write-Host "✗ Failed to configure remote: $_" -ForegroundColor Red
    exit 1
}

# Verify remote
Write-Host "[STEP 4] Verifying remote configuration..." -ForegroundColor Yellow
try {
    $remoteInfo = git remote -v
    Write-Host $remoteInfo -ForegroundColor Cyan
    if ($remoteInfo -match "origin.*$repositoryUrl") {
        Write-Host "✓ Remote correctly configured" -ForegroundColor Green
    }
} catch {
    Write-Host "✗ Failed to verify remote: $_" -ForegroundColor Red
}

# Step 5: Check if .gitignore exists
Write-Host "[STEP 5] Checking .gitignore file..." -ForegroundColor Yellow
if (Test-Path ".gitignore") {
    Write-Host "✓ .gitignore file already exists" -ForegroundColor Green
} else {
    Write-Host "  Creating .gitignore file..." -ForegroundColor Cyan
    $gitignoreContent = @"
__pycache__/
*.pyc
*.pyo
*.pkl
.env
uploads/*
!uploads/.gitkeep
results/accuracy_report.txt
.vscode/
.idea/
*.log
"@
    $gitignoreContent | Out-File -FilePath ".gitignore" -Encoding UTF8
    Write-Host "✓ .gitignore file created" -ForegroundColor Green
}

# Step 6: Stage all files
Write-Host "[STEP 6] Staging all files..." -ForegroundColor Yellow
try {
    git add .
    Write-Host "✓ All files staged" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to stage files: $_" -ForegroundColor Red
    exit 1
}

# Step 7: Check status
Write-Host "[STEP 7] Checking git status..." -ForegroundColor Yellow
try {
    $status = git status --short
    if ($Verbose) {
        Write-Host $status -ForegroundColor Cyan
    }
    $changeCount = ($status -split "`n" | Where-Object { $_ -match "^\s*[AMU]" }).Count
    Write-Host "✓ $changeCount file(s) ready to commit" -ForegroundColor Green
} catch {
    Write-Host "⚠ Could not check status: $_" -ForegroundColor Yellow
}

# Step 8: Create commit
Write-Host "[STEP 8] Creating commit..." -ForegroundColor Yellow
try {
    git commit -m $commitMessage
    Write-Host "✓ Commit created successfully" -ForegroundColor Green
} catch {
    if ($_ -match "nothing to commit") {
        Write-Host "⚠ Nothing new to commit (all changes already committed)" -ForegroundColor Yellow
    } else {
        Write-Host "✗ Failed to create commit: $_" -ForegroundColor Red
        exit 1
    }
}

# Step 9: Get commit info
Write-Host "[STEP 9] Getting commit information..." -ForegroundColor Yellow
try {
    $commitHash = git rev-parse HEAD
    $commitHashShort = $commitHash.Substring(0, 7)
    Write-Host "✓ Commit hash: $commitHashShort" -ForegroundColor Green
} catch {
    Write-Host "⚠ Could not retrieve commit hash" -ForegroundColor Yellow
    $commitHashShort = "Unknown"
}

# Step 10: Push to GitHub
Write-Host "[STEP 10] Pushing to GitHub..." -ForegroundColor Yellow

$pushSuccess = $false

# Try pushing to main
Write-Host "  Attempting main branch..." -ForegroundColor Cyan
try {
    git push -u origin main 2>&1 | ForEach-Object { 
        if ($Verbose) { Write-Host $_ -ForegroundColor Cyan }
    }
    $pushSuccess = $true
    Write-Host "✓ Successfully pushed to main branch" -ForegroundColor Green
} catch {
    Write-Host "  Main branch push failed, trying master..." -ForegroundColor Yellow
    
    # Try pushing to master
    try {
        git push -u origin master 2>&1 | ForEach-Object { 
            if ($Verbose) { Write-Host $_ -ForegroundColor Cyan }
        }
        $pushSuccess = $true
        Write-Host "✓ Successfully pushed to master branch" -ForegroundColor Green
    } catch {
        if ($Force) {
            Write-Host "  Attempting force push to main..." -ForegroundColor Yellow
            try {
                git push -u origin main --force 2>&1 | ForEach-Object { 
                    if ($Verbose) { Write-Host $_ -ForegroundColor Cyan }
                }
                $pushSuccess = $true
                Write-Host "✓ Successfully force-pushed to main branch" -ForegroundColor Green
            } catch {
                Write-Host "✗ Force push failed: $_" -ForegroundColor Red
            }
        } else {
            Write-Host "✗ Push failed. Try with -Force flag if you're the repository owner." -ForegroundColor Red
        }
    }
}

# Step 11: Final verification
Write-Host "[STEP 11] Final verification..." -ForegroundColor Yellow
try {
    $logInfo = git log --oneline -1
    Write-Host "✓ Latest commit: $logInfo" -ForegroundColor Green
} catch {
    Write-Host "⚠ Could not verify latest commit" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PUSH SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Project:        BhoomiAI" -ForegroundColor White
Write-Host "Location:       $projectPath" -ForegroundColor White
Write-Host "Repository:     $repositoryUrl" -ForegroundColor White
Write-Host "Commit Hash:    $commitHashShort" -ForegroundColor White
Write-Host "Commit Message: $commitMessage" -ForegroundColor White

if ($pushSuccess) {
    Write-Host "Status:         ✓ SUCCESSFULLY PUSHED" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Visit: $repositoryUrl" -ForegroundColor White
    Write-Host "2. Verify files are present" -ForegroundColor White
    Write-Host "3. Check the commit history" -ForegroundColor White
} else {
    Write-Host "Status:         ✗ PUSH FAILED" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Cyan
    Write-Host "1. Check internet connection" -ForegroundColor White
    Write-Host "2. Verify GitHub credentials" -ForegroundColor White
    Write-Host "3. Ensure you have write access to the repository" -ForegroundColor White
    Write-Host "4. Check SSH/HTTPS configuration" -ForegroundColor White
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
