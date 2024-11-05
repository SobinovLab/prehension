@echo off
setlocal

rem Define the commit hash
set "commit_hash=690fa2588a1c3bef37539363ce45304d81671bed"

rem Loop over each line in files.txt and checkout the specific version
for /f "delims=" %%f in (cr_local/files.txt) do (
    rem echo Checking out %%f from commit %commit_hash%
    rem git checkout %commit_hash% -- %%f
)

echo Done!
