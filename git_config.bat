@echo off
cd /d "c:\Users\satye\OneDrive\Desktop\krishiai_project"
echo === Step 1: Current Directory ===
cd
echo.

echo === Step 2: Configure Git User ===
git config user.name "Satyendra000"
echo Git user.name set
git config user.email "satyendra000@users.noreply.github.com"
echo Git user.email set
echo.

echo === Step 3: Initialize Git (if needed) ===
git init
echo.

echo === Step 4: Check Remote ===
git remote -v
echo.

echo === Step 5: Check if origin exists ===
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo Adding origin remote...
    git remote add origin https://github.com/karan-nigam96/Bhoomi_AI.git
    echo Origin added
) else (
    echo Origin already exists
)
echo.

echo === Step 6: Verify Remote ===
git remote -v
echo.

echo === Step 7: Check Git Status ===
git status
echo.

echo === Configuration Complete ===
pause
