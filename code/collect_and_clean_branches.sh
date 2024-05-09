# merge all branches created by cluster jobs
git merge -m "Merge results from job cluster" $(git branch -l | grep 'job-' | tr -d ' ')
# delete those branches
git branch -d `git branch --list 'job*'`

