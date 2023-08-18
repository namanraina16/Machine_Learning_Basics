clear
git status
echo -n "Enter files to git add (spaces between multiple files): "
read input
echo "git add $input..." 
git add $input
echo -n "Enter commit message: "
read commit
command="git commit -m \"$commit\""
eval $command
echo "Commit done!"
echo "Pushing to main..."
git push
