@echo off
setlocal enabledelayedexpansion

cd /d c:\Users\satye\OneDrive\Desktop\krishiai_project

echo.
echo ===== Step 1: Check if git repository exists =====
if exist .git (
    echo Directory is already a git repository
) else (
    echo Initializing git repository...
    git init
    if !errorlevel! equ 0 (
        echo Git repository initialized successfully
    ) else (
        echo ERROR: Failed to initialize git repository
        exit /b 1
    )
)
echo.

echo ===== Step 2: Check git status =====
git status
echo.

echo ===== Step 3: Check remote configuration =====
echo Current remote configuration:
git remote -v
echo.

echo ===== Step 4: Configure remote origin =====
for /f %%i in ('git remote -v ^| find "origin" ^| find "fetch"') do set remote_exists=1

if defined remote_exists (
    echo Updating existing remote origin...
    git remote set-url origin https://github.com/karan-nigam96/Bhoomi_AI.git
    if !errorlevel! equ 0 (
        echo Remote URL updated successfully
    ) else (
        echo ERROR: Failed to update remote URL
        exit /b 1
    )
) else (
    echo Adding remote origin...
    git remote add origin https://github.com/karan-nigam96/Bhoomi_AI.git
    if !errorlevel! equ 0 (
        echo Remote origin added successfully
    ) else (
        echo ERROR: Failed to add remote
        exit /b 1
    )
)
echo.

echo ===== Step 5: Create .gitignore file =====
(
    echo __pycache__/
    echo *.pyc
    echo *.pyo
    echo *.pkl
    echo .env
    echo uploads/*
    echo !uploads/.gitkeep
    echo results/accuracy_report.txt
    echo .vscode/
    echo .idea/
    echo *.log
) > .gitignore
echo .gitignore created successfully
echo.

echo ===== Step 6: Stage all files =====
git add .
if !errorlevel! equ 0 (
    echo Files staged successfully
) else (
    echo ERROR: Failed to stage files
    exit /b 1
)
echo.

echo ===== Step 7: Check files to be committed =====
git status
echo.

echo ===== Step 8: Create commit =====
git commit -m "Complete BhoomiAI upgrade: Season and Agro-Zone features integrated"
if !errorlevel! equ 0 (
    echo Commit created successfully
    echo Getting commit hash...
    for /f %%i in ('git rev-parse HEAD') do set commit_hash=%%i
    echo Commit hash: !commit_hash!
) else (
    echo ERROR: Failed to create commit
    exit /b 1
)
echo.

echo ===== Step 9: Verify remote URL =====
git remote -v
echo.

echo ===== Step 10: Push to main branch =====
echo Attempting to push to main branch...
git push -u origin main
if !errorlevel! equ 0 (
    echo PUSH SUCCESSFUL to main branch!
    goto :success
)

echo Attempting to push to master branch...
git push -u origin master
if !errorlevel! equ 0 (
    echo PUSH SUCCESSFUL to master branch!
    goto :success
)

echo ERROR: Failed to push to both main and master branches
exit /b 1

:success
echo.
echo ===== FINAL STATUS REPORT =====
echo Commit Hash: !commit_hash!
echo Repository URL: https://github.com/karan-nigam96/Bhoomi_AI.git
echo.
echo Getting pushed files count...
git log --oneline -1
echo.
echo ===== PUSH COMPLETE =====
exit /b 0
