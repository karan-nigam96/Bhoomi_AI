@echo off
REM GitHub Push Script for BhoomiAI
REM This script automates the process of pushing BhoomiAI to GitHub

setlocal enabledelayedexpansion

REM Project configuration
set PROJECT_PATH=c:\Users\satye\OneDrive\Desktop\krishiai_project
set REPOSITORY_URL=https://github.com/karan-nigam96/Bhoomi_AI.git
set COMMIT_MESSAGE=Complete BhoomiAI upgrade: Season and Agro-Zone features integrated

REM Color codes (escape sequences)
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "CYAN=[96m"
set "RESET=[0m"

cls
echo.
echo ========================================
echo BhoomiAI GitHub Push Script
echo ========================================
echo.

REM Step 1: Change to project directory
echo [STEP 1] Navigating to project directory...
cd /d "%PROJECT_PATH%" || (
    echo Failed to change directory
    exit /b 1
)
echo Project directory: %cd%
echo.

REM Step 2: Check if git repository exists
echo [STEP 2] Checking git repository...
if exist .git (
    echo Directory is already a git repository
) else (
    echo Initializing git repository...
    git init
    if !errorlevel! equ 0 (
        echo Git repository initialized
    ) else (
        echo Failed to initialize git
        exit /b 1
    )
)
echo.

REM Step 3: Configure remote origin
echo [STEP 3] Configuring remote origin...
git remote -v >nul 2>&1
if !errorlevel! equ 0 (
    REM Check if origin exists
    git remote -v | find "origin" >nul 2>&1
    if !errorlevel! equ 0 (
        echo Updating existing remote origin...
        git remote set-url origin "%REPOSITORY_URL%"
    ) else (
        echo Adding new remote origin...
        git remote add origin "%REPOSITORY_URL%"
    )
) else (
    echo Adding new remote origin...
    git remote add origin "%REPOSITORY_URL%"
)
echo Remote configured successfully
echo.

REM Step 4: Verify remote
echo [STEP 4] Verifying remote configuration...
git remote -v
echo.

REM Step 5: Check if .gitignore exists
echo [STEP 5] Checking .gitignore file...
if exist .gitignore (
    echo .gitignore file already exists
) else (
    echo Creating .gitignore file...
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
    echo .gitignore created
)
echo.

REM Step 6: Stage all files
echo [STEP 6] Staging all files...
git add .
echo Files staged
echo.

REM Step 7: Check status
echo [STEP 7] Checking git status...
git status --short
echo.

REM Step 8: Create commit
echo [STEP 8] Creating commit...
git commit -m "%COMMIT_MESSAGE%"
if !errorlevel! neq 0 (
    if !errorlevel! equ 1 (
        echo Note: Nothing new to commit
    ) else (
        echo Failed to create commit
        exit /b 1
    )
)
echo.

REM Step 9: Get commit info
echo [STEP 9] Getting commit information...
for /f %%i in ('git rev-parse HEAD') do set COMMIT_HASH=%%i
set COMMIT_SHORT=%COMMIT_HASH:~0,7%
echo Commit hash: %COMMIT_SHORT%
echo.

REM Step 10: Push to GitHub
echo [STEP 10] Pushing to GitHub...
echo Attempting main branch...
git push -u origin main >nul 2>&1
if !errorlevel! equ 0 (
    echo Successfully pushed to main branch
    goto :push_success
)

echo Attempting master branch...
git push -u origin master >nul 2>&1
if !errorlevel! equ 0 (
    echo Successfully pushed to master branch
    goto :push_success
)

echo Push failed - trying force push to main...
git push -u origin main --force >nul 2>&1
if !errorlevel! equ 0 (
    echo Successfully force-pushed to main branch
    goto :push_success
)

echo Failed to push
exit /b 1

:push_success
echo.

REM Step 11: Final verification
echo [STEP 11] Final verification...
git log --oneline -1
echo.

REM Summary
echo ========================================
echo PUSH SUMMARY
echo ========================================
echo Project:        BhoomiAI
echo Location:       %PROJECT_PATH%
echo Repository:     %REPOSITORY_URL%
echo Commit Hash:    %COMMIT_SHORT%
echo Status:         SUCCESS
echo ========================================
echo.
echo Next steps:
echo 1. Visit: %REPOSITORY_URL%
echo 2. Verify files are present
echo 3. Check the commit history
echo.

endlocal
exit /b 0
