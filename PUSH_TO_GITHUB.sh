#!/bin/bash

# Script to push ML Pipeline to GitHub
# Usage: bash PUSH_TO_GITHUB.sh

echo "=========================================="
echo "ML Pipeline - GitHub Push Script"
echo "=========================================="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Error: Git repository not initialized!"
    echo "Run: git init"
    exit 1
fi

# Check if remote is set
if ! git remote | grep -q "origin"; then
    echo "GitHub remote not set yet."
    echo ""
    echo "Please create a repository on GitHub first, then run:"
    echo ""
    echo "  git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git"
    echo ""
    echo "Or if using SSH:"
    echo ""
    echo "  git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git"
    echo ""
    read -p "Have you created the GitHub repository? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Please create the repository first, then run this script again."
        exit 1
    fi
    
    read -p "Enter your GitHub username: " GITHUB_USER
    read -p "Enter your repository name: " REPO_NAME
    read -p "Use SSH? (y/n) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git remote add origin "git@github.com:${GITHUB_USER}/${REPO_NAME}.git"
    else
        git remote add origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
    fi
    
    echo "Remote added: $(git remote get-url origin)"
fi

# Check current branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ] && [ "$CURRENT_BRANCH" != "master" ]; then
    echo "Current branch: $CURRENT_BRANCH"
    read -p "Rename to 'main'? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git branch -M main
        CURRENT_BRANCH="main"
    fi
fi

# Show status
echo ""
echo "Current status:"
git status --short | head -10
echo ""

# Ask for confirmation
read -p "Push to GitHub? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Push to GitHub
echo ""
echo "Pushing to GitHub..."
git push -u origin "$CURRENT_BRANCH"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Success! Code pushed to GitHub."
    echo "=========================================="
    echo ""
    echo "Repository URL: $(git remote get-url origin)"
    echo ""
    echo "Important reminders:"
    echo "  - dev.yml is NOT committed (contains passwords)"
    echo "  - Users should copy dev.yml.example to dev.yml"
    echo "  - Update passwords in dev.yml before use"
    echo ""
else
    echo ""
    echo "Error: Push failed!"
    echo "Check your GitHub credentials and repository permissions."
    exit 1
fi

