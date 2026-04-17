#!/usr/bin/env python3
"""
GitHub Push Script for BhoomiAI
This script automates the process of pushing BhoomiAI to GitHub
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Configuration
PROJECT_PATH = r"c:\Users\satye\OneDrive\Desktop\krishiai_project"
REPOSITORY_URL = "https://github.com/karan-nigam96/Bhoomi_AI.git"
COMMIT_MESSAGE = "Complete BhoomiAI upgrade: Season and Agro-Zone features integrated"

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

def print_header(text):
    print(f"\n{Colors.CYAN}{'='*40}{Colors.RESET}")
    print(f"{Colors.CYAN}{text}{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*40}{Colors.RESET}\n")

def print_step(step_num, text):
    print(f"{Colors.YELLOW}[STEP {step_num}] {text}{Colors.RESET}")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def run_command(cmd, check=True):
    """Execute a shell command and return output"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=PROJECT_PATH,
            capture_output=True,
            text=True
        )
        if check and result.returncode != 0:
            return False, result.stderr
        return True, result.stdout
    except Exception as e:
        return False, str(e)

def main():
    # Print header
    print_header("BhoomiAI GitHub Push Script")
    
    # Step 1: Check project directory
    print_step(1, "Checking project directory...")
    if not os.path.isdir(PROJECT_PATH):
        print_error(f"Project directory not found: {PROJECT_PATH}")
        return False
    print_success(f"Project directory found: {PROJECT_PATH}")
    
    # Change to project directory
    os.chdir(PROJECT_PATH)
    
    # Step 2: Check if git repository exists
    print_step(2, "Checking git repository...")
    git_dir = os.path.join(PROJECT_PATH, ".git")
    if os.path.isdir(git_dir):
        print_success("Directory is already a git repository")
    else:
        print("Initializing git repository...")
        success, output = run_command("git init")
        if success:
            print_success("Git repository initialized")
        else:
            print_error(f"Failed to initialize git: {output}")
            return False
    
    # Step 3: Configure remote origin
    print_step(3, "Configuring remote origin...")
    success, output = run_command("git remote -v", check=False)
    
    if "origin" in output:
        print("Updating existing remote origin...")
        success, output = run_command(f'git remote set-url origin "{REPOSITORY_URL}"')
        if success:
            print_success("Remote URL updated")
        else:
            print_error(f"Failed to update remote: {output}")
            return False
    else:
        print("Adding new remote origin...")
        success, output = run_command(f'git remote add origin "{REPOSITORY_URL}"')
        if success:
            print_success("Remote origin added")
        else:
            print_error(f"Failed to add remote: {output}")
            return False
    
    # Step 4: Verify remote
    print_step(4, "Verifying remote configuration...")
    success, output = run_command("git remote -v", check=False)
    print(f"{Colors.CYAN}{output}{Colors.RESET}")
    
    # Step 5: Create/check .gitignore
    print_step(5, "Checking .gitignore file...")
    gitignore_path = os.path.join(PROJECT_PATH, ".gitignore")
    if os.path.exists(gitignore_path):
        print_success(".gitignore file already exists")
    else:
        print("Creating .gitignore file...")
        gitignore_content = """__pycache__/
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
"""
        try:
            with open(gitignore_path, 'w') as f:
                f.write(gitignore_content)
            print_success(".gitignore file created")
        except Exception as e:
            print_error(f"Failed to create .gitignore: {e}")
            return False
    
    # Step 6: Stage all files
    print_step(6, "Staging all files...")
    success, output = run_command("git add .")
    if success:
        print_success("All files staged")
    else:
        print_error(f"Failed to stage files: {output}")
        return False
    
    # Step 7: Check status
    print_step(7, "Checking git status...")
    success, output = run_command("git status --short", check=False)
    if output:
        print(f"{Colors.CYAN}{output}{Colors.RESET}")
    else:
        print("No changes to stage")
    
    # Step 8: Create commit
    print_step(8, "Creating commit...")
    success, output = run_command(f'git commit -m "{COMMIT_MESSAGE}"', check=False)
    if "nothing to commit" in output:
        print_warning("Nothing new to commit")
    elif success:
        print_success("Commit created successfully")
    else:
        print_error(f"Failed to create commit: {output}")
        return False
    
    # Step 9: Get commit info
    print_step(9, "Getting commit information...")
    success, commit_hash = run_command("git rev-parse HEAD", check=False)
    if success:
        commit_short = commit_hash.strip()[:7]
        print_success(f"Commit hash: {commit_short}")
    else:
        print_warning("Could not retrieve commit hash")
        commit_short = "Unknown"
    
    # Step 10: Push to GitHub
    print_step(10, "Pushing to GitHub...")
    
    push_success = False
    
    # Try main
    print("Attempting main branch...")
    success, output = run_command("git push -u origin main", check=False)
    if success:
        print_success("Successfully pushed to main branch")
        push_success = True
    else:
        # Try master
        print("Attempting master branch...")
        success, output = run_command("git push -u origin master", check=False)
        if success:
            print_success("Successfully pushed to master branch")
            push_success = True
        else:
            # Try force push
            print("Attempting force push to main...")
            success, output = run_command("git push -u origin main --force", check=False)
            if success:
                print_success("Successfully force-pushed to main branch")
                push_success = True
            else:
                print_error(f"Push failed: {output}")
    
    if not push_success:
        return False
    
    # Step 11: Final verification
    print_step(11, "Final verification...")
    success, output = run_command("git log --oneline -1", check=False)
    if success:
        print_success(f"Latest commit: {output.strip()}")
    
    # Summary
    print_header("PUSH SUMMARY")
    print(f"{Colors.WHITE}Project:        BhoomiAI{Colors.RESET}")
    print(f"{Colors.WHITE}Location:       {PROJECT_PATH}{Colors.RESET}")
    print(f"{Colors.WHITE}Repository:     {REPOSITORY_URL}{Colors.RESET}")
    print(f"{Colors.WHITE}Commit Hash:    {commit_short}{Colors.RESET}")
    print(f"{Colors.WHITE}Commit Message: {COMMIT_MESSAGE}{Colors.RESET}")
    print(f"{Colors.GREEN}Status:         ✓ SUCCESSFULLY PUSHED{Colors.RESET}")
    
    print("\n" + f"{Colors.CYAN}Next steps:{Colors.RESET}")
    print(f"1. Visit: {REPOSITORY_URL}")
    print("2. Verify files are present")
    print("3. Check the commit history")
    
    print("\n" + "="*40 + "\n")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)
